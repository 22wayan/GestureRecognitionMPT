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


# So viele Punkte soll am Ende jede Geste haben.
# Warum: Manche malen schnell (wenige Punkte), manche langsam (viele Punkte).
# Wenn alle Gesten gleich viele Punkte haben, ist das Tempo egal und das Modell
# erkennt sie besser.
RESAMPLE_LENGTH = 48


def _resample(traj, n):
    """Bringt die Punkte einer Geste auf genau n Punkte (gleichmaessig verteilt).

    Idee wie ein Bild auf eine feste Groesse ziehen: eine schnell und eine langsam
    gemalte Geste haben danach gleich viele Punkte und sehen fuer das Modell gleich aus.
    """
    # Zu kurz oder schon die richtige Anzahl? Dann einfach so lassen.
    if n is None or len(traj) < 2 or len(traj) == n:
        return traj
    # Die alten Punkte gleichmaessig auf 0..1 legen, die neuen Punkte auch auf 0..1.
    alte_stellen = np.linspace(0, 1, len(traj))
    neue_stellen = np.linspace(0, 1, n)
    # np.interp rechnet die Zwischenwerte aus -- fuer x und y getrennt.
    x = np.interp(neue_stellen, alte_stellen, traj[:, 0])
    y = np.interp(neue_stellen, alte_stellen, traj[:, 1])
    # x und y wieder nebeneinander zu einer Punkte-Liste zusammensetzen.
    return np.column_stack([x, y])


def _to_features(traj, resample_length=RESAMPLE_LENGTH):
    """Macht aus den rohen Punkten die Zahlen, die das Modell zum Lernen braucht.

    Drei einfache Schritte nacheinander:
      1. _resample     -> alle Gesten auf gleich viele Punkte bringen
      2. _normalize    -> Lage und Groesse angleichen (egal wo/wie gross gemalt)
      3. _add_velocity -> Geschwindigkeit als dritte Spalte anhaengen
    Ergebnis: eine Liste mit (x, y, geschwindigkeit) pro Punkt.

    Wichtig: Training UND Live rufen genau diese Funktion auf. So haben beide
    immer das gleiche Format und koennen nie auseinanderlaufen.
    """
    traj = _resample(traj, resample_length)
    traj = _normalize(traj)
    traj = _add_velocity(traj)
    return traj


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
        # Manche Frames sind None (z.B. wird das stop()-Ergebnis des Detectors
        # mitgespeichert). Solche behandeln wir wie "keine Hand erkannt".
        if not isinstance(frame, dict):
            points.append(None)
            continue

        det = frame.get("detector")
        point = None  # Standard: in diesem Frame keine Hand gefunden

        # Es gibt zwei Aufnahme-Formate -- wir unterstuetzen beide:
        if hasattr(det, "hand_landmarks"):
            # 1) Alte Aufnahmen: rohes MediaPipe-Objekt (hat .hand_landmarks)
            if len(det.hand_landmarks) > 0:
                lm = det.hand_landmarks[0][finger_idx]
                point = [lm.x, lm.y]
        elif isinstance(det, dict):
            # 2) Neue Aufnahmen: einfaches Dict {"hands": [{"landmarks": [...]}]}
            hands = det.get("hands", [])
            if hands:
                lm = hands[0]["landmarks"][finger_idx]
                point = [lm["x"], lm["y"]]

        points.append(point)

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

            traj = _to_features(traj)
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


def _record_one_take() -> str | None:
    """
    Nimmt eine einzelne Geste auf und gibt den Pfad der Temp-Datei zurueck.

    Startet ``main.py`` im Record-Modus als Subprocess und lenkt die Aufnahme
    in eine temporaere Datei. Die leere Temp-Datei wird vorher geloescht, damit
    ihr blosses Vorhandensein nach dem Subprocess bedeutet: "Aufnahme war
    erfolgreich".

    Der Aufrufer ist dafuer verantwortlich, die zurueckgegebene Temp-Datei
    entweder an ihren Zielort zu verschieben oder zu loeschen.

    Returns
    -------
    str or None
        Pfad zur Temp-Datei mit der Aufnahme, oder ``None`` wenn keine
        Aufnahme entstanden ist (z. B. bei einem Crash).
    """
    # Temporaere Datei fuer diese eine Aufnahme erstellen.
    # SignalHub schreibt die Aufnahme erst beim sauberen Beenden in diese Datei.
    # Wir loeschen die leere Temp-Datei wieder, damit ihr blosses Vorhandensein
    # spaeter bedeutet: "Aufnahme war erfolgreich".
    fd, temp_file = tempfile.mkstemp(suffix=".pkl")
    os.close(fd)
    os.remove(temp_file)

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
    # etwas schiefgelaufen.
    if not os.path.exists(temp_file):
        return None
    return temp_file


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

        # Eine einzelne Aufnahme machen (Temp-Datei mit Crash-Schutz).
        temp_file = _record_one_take()

        # Crash-Schutz: Wenn keine Datei entstanden ist, ist beim Aufnehmen
        # etwas schiefgelaufen. Wir speichern dann nichts und versuchen es erneut.
        if temp_file is None:
            print("Es wurde keine Aufnahme erzeugt. Bitte erneut versuchen.")
            continue

        try:
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


# Buchstaben A-Z, die im Team-Workflow aufgenommen werden.
ALPHABET = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def _person_takes(label_dir: Path, letter: str, person: str) -> int:
    """
    Zaehlt, wie viele Takes diese Person fuer einen Buchstaben schon hat.

    Beruecksichtigt nummerierte Takes (``<L>-<person>-<n>.pkl``) und – aus
    Abwaertskompatibilitaet – die alte Einzeldatei (``<L>-<person>.pkl``).
    """
    numbered = list(label_dir.glob(f"{letter}-{person}-*.pkl"))
    legacy = label_dir / f"{letter}-{person}.pkl"
    return len(numbered) + (1 if legacy.exists() else 0)


def _next_person_take_path(label_dir: Path, letter: str, person: str) -> Path:
    """
    Findet den naechsten freien Take-Pfad ``<L>-<person>-<n>.pkl``.

    Zaehlt ab 1 hoch, bis ein noch nicht vergebener Name gefunden ist, damit
    keine bestehende Aufnahme ueberschrieben wird.
    """
    n = 1
    while (label_dir / f"{letter}-{person}-{n}.pkl").exists():
        n += 1
    return label_dir / f"{letter}-{person}-{n}.pkl"


def _letter_range(start: str | None, end: str | None) -> list[str]:
    """
    Baut die Liste der aufzunehmenden Buchstaben aus optionalem Start/Ende.

    ``start``/``end`` sind einzelne Buchstaben (A-Z, Gross-/Kleinschreibung
    egal). Fehlt ``start``, wird bei "A" begonnen; fehlt ``end``, bis "Z".
    """
    s = (start or "A").strip().upper()
    e = (end or "Z").strip().upper()
    for name, val in (("start", s), ("end", e)):
        if len(val) != 1 or not ("A" <= val <= "Z"):
            raise ValueError(f"{name} muss ein einzelner Buchstabe A-Z sein, war '{val}'.")
    if s > e:
        raise ValueError(f"start ({s}) liegt hinter end ({e}).")
    return [chr(c) for c in range(ord(s), ord(e) + 1)]


def collect_alphabet(
    person: str,
    times: int = 15,
    recordings_dir: str | Path = "recordings",
    start: str | None = None,
    end: str | None = None,
):
    """
    Gefuehrter Aufnahme-Durchlauf: jeder Buchstabe A-Z mehrfach pro Person.

    Fuehrt eine Person automatisch durch das komplette Alphabet und nimmt pro
    Buchstabe ``times`` Gesten auf. Die Aufnahmen landen direkt unter
    ``recordings/<Buchstabe>/<Buchstabe>-<person>-<n>.pkl`` -- also genau dort,
    wo :func:`dataset_building` standardmaessig liest (es liest alle ``*.pkl``
    eines Ordners). Damit fliessen die neuen Aufnahmen ohne weitere Schritte
    ins Training.

    Resume-Faehigkeit
    -----------------
    Pro Buchstabe wird gezaehlt, wie viele Takes diese Person bereits hat;
    aufgenommen wird nur, bis ``times`` erreicht ist. Der Durchlauf kann also
    jederzeit abgebrochen und spaeter fortgesetzt werden, ohne etwas zu
    ueberschreiben.

    Parameters
    ----------
    person : str
        Name oder Kuerzel der aufnehmenden Person (z. B. "arian"). Wird Teil
        des Dateinamens, damit die Diversitaet nachvollziehbar bleibt.
    times : int
        Wie viele Takes pro Buchstabe diese Person aufnehmen soll (Standard: 15).
    recordings_dir : str or Path
        Zielverzeichnis mit den Label-Unterordnern (Standard: ``recordings``).
    """
    if times < 1:
        raise ValueError(f"times muss >= 1 sein, war {times}.")
    # Personennamen bereinigen und auf dateinamen-taugliche Zeichen pruefen.
    person = person.strip()
    if not person:
        raise ValueError("Bitte einen nicht-leeren Namen / ein Kuerzel angeben.")
    if any(c in person for c in r'/\: '):
        raise ValueError(
            f"Ungueltiger Name '{person}': keine Leerzeichen, Slashes oder "
            "Doppelpunkte erlauben (wird Teil des Dateinamens)."
        )

    recordings_dir = Path(recordings_dir)

    # Welche Buchstaben werden in diesem Durchlauf aufgenommen?
    letters = _letter_range(start, end)

    # Kurze Anleitung ausgeben
    print("=" * 50)
    print(f"Alphabet-Aufnahme fuer: {person}")
    print(
        f"Es werden die Buchstaben {letters[0]}-{letters[-1]} "
        f"({len(letters)} Stueck) je {times}x aufgenommen."
    )
    print("Bereits aufgenommene Takes werden uebersprungen.")
    print("Ablauf pro Buchstabe:")
    print("  1. ENTER druecken, dann fuehrt sich das Aufnahme-Fenster.")
    print("  2. Geste ausfuehren und das Fenster schliessen.")
    print("  3. Danach entscheiden: speichern / verwerfen / abbrechen.")
    print("=" * 50)

    saved = 0
    skipped = 0

    aborted = False
    for i, letter in enumerate(letters, start=1):
        label_dir = recordings_dir / letter
        label_dir.mkdir(parents=True, exist_ok=True)

        # Resume: schon vorhandene Takes dieser Person mitzaehlen.
        have_letter = _person_takes(label_dir, letter, person)
        if have_letter >= times:
            print(
                f"[{i}/{len(letters)}] {letter}: bereits {have_letter}/{times} "
                "-> uebersprungen"
            )
            skipped += 1
            continue

        # Pro Buchstabe so lange aufnehmen, bis times Takes erreicht sind
        # (oder der ganze Vorgang abgebrochen wird).
        while have_letter < times and not aborted:
            take_no = have_letter + 1
            print()
            print("#" * 50)
            print(
                f"#   Buchstabe:  {letter}   Take {take_no}/{times}   "
                f"[{i}/{len(letters)}]"
            )
            print("#" * 50)
            input("Druecke ENTER, um die Aufnahme zu starten ...")

            temp_file = _record_one_take()
            if temp_file is None:
                print("Es wurde keine Aufnahme erzeugt. Bitte erneut versuchen.")
                continue

            try:
                print("Was soll mit der Aufnahme passieren?")
                print("  [s] speichern")
                print("  [v] verwerfen (Take wiederholen)")
                print("  [a] abbrechen (ganzen Durchlauf beenden)")
                choice = input("Deine Wahl: ").strip().lower()

                if choice == "s":
                    dest = _next_person_take_path(label_dir, letter, person)
                    shutil.move(temp_file, dest)
                    saved += 1
                    have_letter += 1
                    print(f"Gespeichert: {dest}")
                elif choice == "a":
                    print("Durchlauf abgebrochen.")
                    aborted = True
                else:
                    print("Aufnahme verworfen.")
            finally:
                # Aufraeumen: eine evtl. noch vorhandene Temp-Datei loeschen.
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        if aborted:
            break

    # Abschluss: wie viele Buchstaben hat diese Person nun vollstaendig (times Takes)?
    complete = sum(
        1 for letter in letters
        if _person_takes(recordings_dir / letter, letter, person) >= times
    )
    print()
    print("=" * 50)
    print(f"Fertig. Neu gespeichert: {saved}, uebersprungen: {skipped}.")
    print(
        f"'{person}' hat jetzt {complete}/{len(letters)} Buchstaben "
        f"aus {letters[0]}-{letters[-1]} vollstaendig ({times} Takes)."
    )
    if complete < len(letters):
        print("Tipp: Skript erneut starten, um die fehlenden Takes zu ergaenzen.")
    print("=" * 50)


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

    # Sequenzen sammeln und dabei solche mit NaN-Werten ueberspringen.
    # NaN entsteht bei Frames ohne erkannte Hand mitten in der Aufnahme.
    # Solche Werte landen sonst ungefiltert im Trainings-Array, womit das
    # HMM-Training nicht zurechtkommt.
    sequences: list[np.ndarray] = []
    labels: list[str] = []
    nan_skipped: dict[str, int] = defaultdict(int)
    for label, trajectories in dataset.items():
        for traj in trajectories:
            if np.isnan(traj).any():
                nan_skipped[label] += 1
                continue
            sequences.append(traj)
            labels.append(label)

    if nan_skipped:
        total_skipped = sum(nan_skipped.values())
        logger.info(
            "%d Sequenz(en) wegen NaN-Werten uebersprungen: %s",
            total_skipped,
            dict(nan_skipped),
        )

    if not sequences:
        raise ValueError("Keine gueltigen Aufnahmen gefunden.")

    # Stratifizierter Split braucht pro Klasse mindestens zwei Aufnahmen,
    # damit jede Klasse in Train- UND Test-Set vertreten sein kann.
    # Sonst bricht train_test_split mit einer schwer verstaendlichen
    # Meldung ab -- deshalb hier vorab pruefen und klar melden.
    class_counts = Counter(labels)
    too_few = {
        label: count for label, count in class_counts.items() if count < 2
    }
    if too_few:
        details = ", ".join(
            f"'{label}': {count} Aufnahme(n)" for label, count in sorted(too_few.items())
        )
        raise ValueError(
            "Fuer einen stratifizierten Train/Test-Split werden pro Klasse "
            "mindestens 2 gueltige Aufnahmen benoetigt. Zu wenige Aufnahmen "
            f"bei: {details}. Bitte mehr Aufnahmen erstellen oder die "
            "Bereinigungs-Parameter (min_length, max_jump) lockern."
        )

    # Zusaetzlich zur Pro-Klasse-Pruefung braucht der stratifizierte Split von
    # sklearn, dass BEIDE Seiten (Train und Test) mindestens so viele Sequenzen
    # bekommen wie es Klassen gibt. Sonst bricht train_test_split mit einer
    # schwer verstaendlichen Meldung ab -- typisch waehrend der Datensammlung:
    # viele Klassen (A-Z) mit noch wenigen Aufnahmen pro Klasse.
    n_classes = len(class_counts)
    n_test = int(np.ceil(test_size * len(sequences)))
    n_train = len(sequences) - n_test
    if min(n_train, n_test) < n_classes:
        needed = int(np.ceil(n_classes / min(test_size, 1 - test_size)))
        raise ValueError(
            f"Zu wenige Aufnahmen fuer einen stratifizierten Train/Test-Split "
            f"ueber {n_classes} Klassen (Train {n_train}, Test {n_test} Sequenzen; "
            f"beide muessen >= {n_classes} sein). Bitte insgesamt mindestens "
            f"{needed} gueltige Aufnahmen erstellen oder test_size anpassen."
        )

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