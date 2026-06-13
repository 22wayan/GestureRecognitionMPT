import os
import sys
import shutil
import tempfile
import subprocess
import pickle
import logging
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def _extract_trajectory(recording: dict, finger_idx: int) -> np.ndarray | None:
    """
    Extrahiert die (x, y)-Trajektorie eines Fingers aus einer Aufnahme.

    Hand-lose Frames am Anfang und Ende werden automatisch abgeschnitten.
    Frames ohne erkannte Hand mitten in der Sequenz bleiben erhalten —
    sie werden als ``nan`` eingetragen, damit der Index erhalten bleibt.

    Parameters
    ----------
    recording : dict
        Inhalt einer .pkl-Aufnahmedatei (Schlüssel ``"detector"``).
    finger_idx : int
        MediaPipe-Landmark-Index des zu verfolgenden Fingers (z.B. 8 = Zeigefingerspitze).

    Returns
    -------
    np.ndarray or None
        Array der Form (N, 2) mit (x, y)-Koordinaten, oder ``None`` wenn
        nach dem Trimmen keine Frames übrig bleiben.
    """
    frames = recording.get("detector", [])

    points = []
    for frame in frames:
        det = frame.get("detector")
        if det and len(det.hand_landmarks) > 0:
            lm = det.hand_landmarks[0][finger_idx]
            points.append([lm.x, lm.y])
        else:
            points.append(None)

    # Anfang und Ende ohne Hand abschneiden
    start = 0
    while start < len(points) and points[start] is None:
        start += 1

    end = len(points) - 1
    while end >= start and points[end] is None:
        end -= 1

    trimmed = points[start:end + 1]
    if not trimmed:
        return None

    result = np.array(
        [p if p is not None else [np.nan, np.nan] for p in trimmed],
        dtype=float,
    )
    return result


def _is_outlier(traj: np.ndarray, max_jump: float) -> bool:
    """
    Prüft ob eine Trajektorie unrealistische Frame-zu-Frame-Sprünge enthält.

    Ein Sprung entsteht, wenn MediaPipe kurz die Hand verliert und dann
    an einer völlig anderen Stelle wieder auffindet. Solche Sprünge
    verfälschen das Modell, weil sie keine echte Bewegung darstellen.

    Der Schwellenwert ``max_jump`` ist in normalisierten Bildkoordinaten
    angegeben (0.0–1.0). Werte über ~0.15 sind in normalen Gesten nicht
    erreichbar — selbst schnelle Bewegungen liegen deutlich darunter
    (95. Perzentil in den Aufnahmen: ~0.04, Maximum: ~0.11).

    Parameters
    ----------
    traj : np.ndarray
        Trajektorie der Form (N, 2).
    max_jump : float
        Maximaler erlaubter euklidischer Abstand zwischen zwei Frames.

    Returns
    -------
    bool
        ``True`` wenn ein Sprung größer als ``max_jump`` gefunden wurde.
    """
    valid = traj[~np.isnan(traj).any(axis=1)]
    if len(valid) < 2:
        return False
    diffs = np.linalg.norm(np.diff(valid, axis=0), axis=1)
    return bool(np.any(diffs > max_jump))


def _normalize(traj: np.ndarray) -> np.ndarray:
    """
    Zentriert und skaliert eine Trajektorie.

    Konsistent mit der Preprocessor-Logik: Mittelpunkt wird abgezogen,
    dann durch den maximalen Abstand zum Mittelpunkt geteilt. Dadurch
    liegen alle Punkte im Einheitskreis um den Ursprung.

    nan-Werte werden bei der Berechnung ignoriert und unverändert
    weitergegeben.

    Parameters
    ----------
    traj : np.ndarray
        Trajektorie der Form (N, 2).

    Returns
    -------
    np.ndarray
        Normalisierte Trajektorie, gleiche Form wie Eingabe.
    """
    valid_mask = ~np.isnan(traj).any(axis=1)
    valid = traj[valid_mask]

    if len(valid) == 0:
        return traj.copy()

    center = valid.mean(axis=0)
    centered = valid - center
    scale = np.linalg.norm(centered, axis=1).max()

    if scale == 0:
        return traj.copy()

    result = traj.copy()
    result[valid_mask] = centered / scale
    return result


def _add_velocity(traj: np.ndarray) -> np.ndarray:
    """
    Erweitert eine Trajektorie um die Frame-zu-Frame-Geschwindigkeit.

    Konsistent mit der Preprocessor-Logik: Die Geschwindigkeit ist der
    euklidische Abstand zwischen aufeinanderfolgenden Frames, als
    skalarer Wert an jede Zeile angehängt. Der erste Frame bekommt 0.0.

    Parameters
    ----------
    traj : np.ndarray
        Trajektorie der Form (N, 2).

    Returns
    -------
    np.ndarray
        Erweiterte Trajektorie der Form (N, 3) mit Spalten (x, y, velocity).
    """
    velocity = np.zeros((len(traj), 1))
    for i in range(1, len(traj)):
        if not (np.isnan(traj[i]).any() or np.isnan(traj[i - 1]).any()):
            velocity[i] = np.linalg.norm(traj[i] - traj[i - 1])
        else:
            velocity[i] = np.nan
    return np.hstack([traj, velocity])


def clean_recordings(
    recordings_dir: str | Path,
    finger_idx: int = 8,
    min_length: int = 15,
    max_jump: float = 0.15,
) -> dict[str, list[np.ndarray]]:
    """
    Lädt alle Aufnahmen aus einem Verzeichnis und filtert fehlerhafte heraus.

    Erwartet eine Ordnerstruktur wie ``recordings/<label>/<datei>.pkl``.
    Für jede Klasse wird geloggt, wie viele Aufnahmen verworfen wurden
    und warum.

    Parameters
    ----------
    recordings_dir : str or Path
        Pfad zum Verzeichnis mit den Label-Unterordnern.
    finger_idx : int
        MediaPipe-Landmark-Index des Fingers (Standard: 8 = Zeigefingerspitze).
    min_length : int
        Minimale Anzahl valider Frames nach dem Trimmen.
    max_jump : float
        Maximaler erlaubter Frame-zu-Frame-Sprung (normalisierte Koordinaten).

    Returns
    -------
    dict[str, list[np.ndarray]]
        Mapping von Label zu Liste bereinigter, normalisierter Trajektorien
        mit Geschwindigkeits-Feature (Form: (N, 3)).
    """
    recordings_dir = Path(recordings_dir)
    dataset = {}

    for label_dir in sorted(recordings_dir.iterdir()):
        if not label_dir.is_dir():
            continue

        label = label_dir.name
        kept = []
        stats = defaultdict(int)

        for pkl_file in sorted(label_dir.glob("*.pkl")):
            with open(pkl_file, "rb") as f:
                recording = pickle.load(f)

            traj = _extract_trajectory(recording, finger_idx)

            if traj is None or len(traj) < min_length:
                stats["zu kurz"] += 1
                logger.info("[%s] %s verworfen: zu kurz (%s Frames)", label, pkl_file.name, len(traj) if traj is not None else 0)
                continue

            if _is_outlier(traj, max_jump):
                stats["Tracking-Sprung"] += 1
                logger.info("[%s] %s verworfen: Tracking-Sprung erkannt", label, pkl_file.name)
                continue

            traj = _normalize(traj)
            traj = _add_velocity(traj)
            kept.append(traj)

        total = sum(stats.values()) + len(kept)
        logger.info(
            "[%s] %d/%d Aufnahmen behalten | verworfen: %s",
            label,
            len(kept),
            total,
            dict(stats) if stats else "keine",
        )

        dataset[label] = kept

    return dataset


def _next_recording_index(label_dir: str) -> int:
    """
    Sucht die naechste freie Nummer fuer eine Aufnahme.

    Wir starten bei 1 und zaehlen hoch, solange die Datei schon existiert.
    So wird nie eine bestehende Aufnahme ueberschrieben.
    """
    i = 1
    while os.path.exists(os.path.join(label_dir, f"recording_{i}.pkl")):
        i += 1
    return i


def data_labeling(times: int, label: str):
    """
    Aufnahme-Workflow fuer eigene Trainingsdaten.

    Es werden nacheinander Aufnahmen gemacht. Pro Aufnahme entscheidet der
    Benutzer, ob die Aufnahme gespeichert, verworfen oder der ganze Vorgang
    abgebrochen wird. Gespeicherte Aufnahmen landen unter
    ``data/{label}/recording_{i}.pkl``.

    Parameters
    ----------
    times : int
        Wie viele Aufnahmen gespeichert werden sollen.
    label : str
        Name der Geste / Klasse (z. B. "A").
    """

    # Zielordner fuer dieses Label, z. B. data/A
    label_dir = os.path.join("data", label)
    # Ordner anlegen, falls er noch nicht existiert (kein Fehler, wenn schon da)
    os.makedirs(label_dir, exist_ok=True)

    # Kurze Anleitung fuer den Benutzer ausgeben
    print("=" * 50)
    print(f"Aufnahme-Workflow fuer Label: {label}")
    print(f"Es sollen {times} Aufnahme(n) gespeichert werden.")
    print("Ablauf pro Aufnahme:")
    print("  1. Ein Fenster oeffnet sich und nimmt die Geste auf.")
    print("  2. Fenster schliessen, um die Aufnahme zu beenden.")
    print("  3. Danach entscheiden: speichern / verwerfen / abbrechen.")
    print("=" * 50)

    # Zaehlt, wie viele Aufnahmen schon erfolgreich gespeichert wurden
    saved = 0

    # Schleife laeuft, bis die gewuenschte Anzahl gespeichert wurde
    while saved < times:
        print()
        print(f"--- Aufnahme {saved + 1} von {times} (Label: {label}) ---")
        input("Druecke ENTER, um die Aufnahme zu starten ...")

        # Temporaere Datei fuer diese eine Aufnahme erstellen.
        # SignalHub schreibt die Aufnahme erst beim sauberen Beenden in diese Datei.
        # Wir loeschen die leere Temp-Datei wieder, damit ihr blosses Vorhandensein
        # spaeter bedeutet: "Aufnahme war erfolgreich".
        fd, temp_file = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)
        os.remove(temp_file)

        try:
            # main.py als Subprocess starten und die Aufnahme in die Temp-Datei lenken.
            # --mode record sorgt dafuer, dass aufgenommen wird.
            cmd = [
                sys.executable,
                "main.py",
                "--mode",
                "record",
                "--recorder.file",
                temp_file,
            ]
            subprocess.run(cmd)

            # Crash-Schutz: Wenn keine Datei entstanden ist, ist beim Aufnehmen
            # etwas schiefgelaufen. Wir speichern dann nichts und versuchen es erneut.
            if not os.path.exists(temp_file):
                print("Es wurde keine Aufnahme erzeugt. Bitte erneut versuchen.")
                continue

            # Benutzer entscheiden lassen, was mit der Aufnahme passiert
            print("Was soll mit der Aufnahme passieren?")
            print("  [s] speichern")
            print("  [v] verwerfen")
            print("  [a] abbrechen")
            choice = input("Deine Wahl: ").strip().lower()

            if choice == "s":
                # Naechste freie Nummer suchen, damit nichts ueberschrieben wird
                index = _next_recording_index(label_dir)
                dest = os.path.join(label_dir, f"recording_{index}.pkl")
                # Temp-Datei an den endgueltigen Ort verschieben (erst jetzt "echte" Daten)
                shutil.move(temp_file, dest)
                saved += 1
                print(f"Gespeichert: {dest}")
            elif choice == "a":
                # Abbrechen: Temp-Datei loeschen und Schleife verlassen
                print("Abgebrochen.")
                break
            else:
                # Alles andere = verwerfen: Temp-Datei loeschen, gleiche Aufnahme wiederholen
                print("Aufnahme verworfen.")
        finally:
            # Aufraeumen: eine evtl. noch vorhandene Temp-Datei loeschen.
            # Das verhindert, dass verworfene oder abgebrochene Aufnahmen Reste hinterlassen.
            if os.path.exists(temp_file):
                os.remove(temp_file)

    # Abschluss-Meldung
    print()
    print(f"Fertig. {saved} Aufnahme(n) fuer Label '{label}' gespeichert.")




def dataset_building(
    output_path: str | Path,
    recordings_dir: str | Path = "recordings",
    finger_idx: int = 8,
    min_length: int = 15,
    max_jump: float = 0.15,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Baut aus den Rohaufnahmen einen Trainingsdatensatz für den HMMClassifier.

    Lädt und bereinigt alle Aufnahmen über :func:`clean_recordings`, fasst
    sie zu ``X``/``y``/``lengths``-Arrays zusammen (das Format, das
    :meth:`HMMClassifier.fit` erwartet) und teilt sie auf Sequenz-Ebene in
    Train- und Test-Set.

    Sequenz-Level-Split (Vermeidung von Data Leakage)
    --------------------------------------------------
    Jede Aufnahme ist eine ganze Geste (Sequenz von Frames). Würde man
    stattdessen einzelne Frames zufällig auf Train/Test verteilen, könnten
    Frames derselben Geste in beiden Sets landen — das Modell würde dann
    quasi "auswendig lernen" und die Testgenauigkeit wäre künstlich hoch.
    Deshalb wird hier auf Höhe ganzer Aufnahmen gesplittet: Jede Sequenz
    landet komplett in genau einem der beiden Sets.

    Stratifizierung
    ----------------
    ``stratify=labels`` sorgt dafür, dass Train- und Test-Set für jede
    Klasse den gleichen Anteil an Aufnahmen enthalten (z.B. bei 10
    Aufnahmen und ``test_size=0.2`` landen pro Klasse 2 im Test-Set).
    Ohne Stratifizierung könnte eine Klasse bei wenigen Aufnahmen rein
    zufällig komplett im Train- oder Test-Set landen.

    Parameters
    ----------
    output_path : str or Path
        Zielpfad für die erzeugte Pickle-Datei.
    recordings_dir : str or Path
        Verzeichnis mit den Rohaufnahmen (``recordings/<label>/*.pkl``).
    finger_idx : int
        MediaPipe-Landmark-Index des verfolgten Fingers.
    min_length : int
        Minimale Sequenzlänge, siehe :func:`clean_recordings`.
    max_jump : float
        Maximaler Frame-zu-Frame-Sprung, siehe :func:`clean_recordings`.
    test_size : float
        Anteil der Aufnahmen pro Klasse, der ins Test-Set wandert.
    random_state : int
        Seed für den Split, für Reproduzierbarkeit.

    Returns
    -------
    dict
        Dictionary mit den Schlüsseln ``X_train``, ``y_train``,
        ``lengths_train``, ``X_test``, ``y_test``, ``lengths_test`` sowie
        ``classes`` (sortierte Liste aller Klassenlabels). ``X_*`` sind
        konkatenierte Feature-Arrays, ``lengths_*`` enthält die Länge jeder
        einzelnen Sequenz darin — direkt nutzbar für
        ``HMMClassifier.fit(X_train, y_train, lengths_train)``.
    """
    dataset = clean_recordings(
        recordings_dir, finger_idx=finger_idx, min_length=min_length, max_jump=max_jump
    )

    sequences: list[np.ndarray] = []
    labels: list[str] = []
    for label, trajectories in dataset.items():
        for traj in trajectories:
            sequences.append(traj)
            labels.append(label)

    if not sequences:
        raise ValueError("Keine gueltigen Aufnahmen gefunden.")

    indices = np.arange(len(sequences))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=labels, random_state=random_state
    )

    def _build_split(idx_list):
        seqs = [sequences[i] for i in idx_list]
        labs = [labels[i] for i in idx_list]
        lengths = [len(seq) for seq in seqs]
        X = np.vstack(seqs)
        y = np.array(labs)
        return X, y, lengths

    X_train, y_train, lengths_train = _build_split(train_idx)
    X_test, y_test, lengths_test = _build_split(test_idx)

    train_counts = Counter(y_train.tolist())
    test_counts = Counter(y_test.tolist())
    for label in sorted(dataset.keys()):
        logger.info(
            "[%s] train: %d, test: %d",
            label,
            train_counts.get(label, 0),
            test_counts.get(label, 0),
        )

    result = {
        "X_train": X_train,
        "y_train": y_train,
        "lengths_train": lengths_train,
        "X_test": X_test,
        "y_test": y_test,
        "lengths_test": lengths_test,
        "classes": sorted(dataset.keys()),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(result, f)

    logger.info("Datensatz gespeichert: %s", output_path)
    return result