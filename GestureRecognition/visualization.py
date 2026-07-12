import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from GestureRecognition.hmmclassifier import HMMClassifier
from GestureRecognition.labeling import (
    _add_velocity,
    _extract_trajectory,
    _is_outlier,
    _normalize,
    _to_features,
    clean_recordings,
    dataset_building,
)

logger = logging.getLogger(__name__)


def _class_colors(classes: list[str]) -> dict[str, tuple]:
    """Weist jeder Klasse eine feste Farbe aus einer Colormap zu."""
    cmap = plt.get_cmap("tab20")
    return {label: cmap(i % cmap.N) for i, label in enumerate(classes)}


def _plot_trajectories(dataset: dict[str, list[np.ndarray]], colors: dict, output_dir: Path) -> None:
    """
    Plot 1: Trajektorien pro Klasse.

    Überlagert die (x, y)-Pfade aller Aufnahmen einer Klasse in einem
    eigenen Subplot. Zeigt, ob die Gesten einer Klasse eine
    wiedererkennbare, konsistente Form haben — und macht Ausreißer
    (z.B. eine Aufnahme mit komplett anderer Form) sichtbar.
    """
    classes = sorted(dataset.keys())
    n_cols = 6
    n_rows = int(np.ceil(len(classes) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, label in zip(axes, classes):
        for traj in dataset[label]:
            ax.plot(traj[:, 0], traj[:, 1], color=colors[label], alpha=0.5, linewidth=1)
            ax.plot(traj[0, 0], traj[0, 1], "o", color=colors[label], markersize=4)
        ax.set_title(label)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        # Bild-/MediaPipe-y waechst nach UNTEN, matplotlib-y nach OBEN. Ohne diese
        # Zeile stuenden die Buchstaben auf dem Kopf (U -> Bogen). invert_yaxis()
        # bringt die Bahn in echte Bildkoordinaten, also aufrecht wie gezeichnet.
        ax.invert_yaxis()

    for ax in axes[len(classes):]:
        ax.axis("off")

    fig.suptitle("Trajektorien pro Klasse (Startpunkt = Punkt)")
    fig.tight_layout()
    fig.savefig(output_dir / "trajectories_per_class.png", dpi=150)
    plt.close(fig)


def _plot_length_histogram(dataset: dict[str, list[np.ndarray]], colors: dict, output_dir: Path) -> None:
    """
    Plot 2: Histogramm der Sequenzlängen pro Klasse.

    Zeigt für jede Klasse die Verteilung der Aufnahmedauer (Anzahl
    Frames). Große Unterschiede in der Länge zwischen oder innerhalb
    von Klassen können auf inkonsistente Ausführung der Geste oder auf
    Aufnahme-Probleme hinweisen.
    """
    classes = sorted(dataset.keys())
    n_cols = 6
    n_rows = int(np.ceil(len(classes) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    all_lengths = [len(traj) for trajs in dataset.values() for traj in trajs]
    bins = np.linspace(min(all_lengths), max(all_lengths), 10) if all_lengths else 10

    for ax, label in zip(axes, classes):
        lengths = [len(traj) for traj in dataset[label]]
        ax.hist(lengths, bins=bins, color=colors[label])
        ax.set_title(label)

    for ax in axes[len(classes):]:
        ax.axis("off")

    fig.suptitle("Verteilung der Sequenzlängen (Frames) pro Klasse")
    fig.tight_layout()
    fig.savefig(output_dir / "sequence_length_histogram.png", dpi=150)
    plt.close(fig)


def _plot_velocity_profiles(dataset: dict[str, list[np.ndarray]], colors: dict, output_dir: Path) -> None:
    """
    Plot 3: Geschwindigkeitsprofile pro Klasse.

    Zeigt für jede Aufnahme die Geschwindigkeit (drittes Feature aus
    `clean_recordings`) über die normalisierte Zeit (0–1), zusammen mit
    der Mittelwert-Trajektorie über alle Aufnahmen der Klasse. Das macht
    typische Bewegungsmuster (z.B. "schnell-langsam-schnell") sichtbar
    und zeigt Aufnahmen, deren Tempo stark vom Rest der Klasse abweicht.
    """
    classes = sorted(dataset.keys())
    n_cols = 6
    n_rows = int(np.ceil(len(classes) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    # Gemeinsames Zeitraster, damit Aufnahmen unterschiedlicher Länge
    # gemittelt werden können.
    n_points = 50
    grid = np.linspace(0, 1, n_points)

    for ax, label in zip(axes, classes):
        resampled = []
        for traj in dataset[label]:
            velocity = traj[:, 2]
            t = np.linspace(0, 1, len(velocity))
            ax.plot(t, velocity, color=colors[label], alpha=0.3, linewidth=1)
            resampled.append(np.interp(grid, t, velocity))

        if resampled:
            mean_velocity = np.mean(resampled, axis=0)
            ax.plot(grid, mean_velocity, color="black", linewidth=2, label="Mittelwert")

        ax.set_title(label)

    for ax in axes[len(classes):]:
        ax.axis("off")

    fig.suptitle("Geschwindigkeitsprofile pro Klasse (normalisierte Zeit)")
    fig.tight_layout()
    fig.savefig(output_dir / "velocity_profiles.png", dpi=150)
    plt.close(fig)


def visualize_dataset(
    recordings_dir: str | Path = "recordings",
    output_dir: str | Path = "plots",
    finger_idx: int = 8,
    min_length: int = 15,
    max_jump: float = 0.15,
) -> dict[str, list[np.ndarray]]:
    """
    Erzeugt Visualisierungen zur Inspektion und Qualitätsprüfung des Datensatzes.

    Lädt die Aufnahmen über :func:`GestureRecognition.labeling.clean_recordings`
    und speichert drei PNG-Dateien unter ``output_dir``:

    - ``trajectories_per_class.png``: (x, y)-Pfade mehrerer Aufnahmen pro
      Klasse überlagert. Zeigt, ob die Gesten einer Klasse eine
      wiedererkennbare, konsistente Form haben.
    - ``sequence_length_histogram.png``: Verteilung der Sequenzlängen pro
      Klasse. Zeigt, wie konsistent die Aufnahmedauer innerhalb einer
      Klasse ist.
    - ``velocity_profiles.png``: Geschwindigkeit über normalisierte Zeit
      pro Aufnahme, zusammen mit der Mittelwert-Trajektorie pro Klasse.
      Zeigt typische Bewegungsmuster und Tempo-Ausreißer.

    Jede Klasse bekommt über alle drei Plots hinweg dieselbe Farbe
    (Colormap ``tab20``).

    Parameters
    ----------
    recordings_dir : str or Path
        Verzeichnis mit den Rohaufnahmen (``recordings/<label>/*.pkl``).
    output_dir : str or Path
        Zielverzeichnis für die PNG-Dateien (wird angelegt, falls nicht
        vorhanden).
    finger_idx : int
        MediaPipe-Landmark-Index des verfolgten Fingers.
    min_length : int
        Minimale Sequenzlänge, siehe :func:`clean_recordings`.
    max_jump : float
        Maximaler Frame-zu-Frame-Sprung, siehe :func:`clean_recordings`.

    Returns
    -------
    dict[str, list[np.ndarray]]
        Das geladene, bereinigte Dataset (wie von :func:`clean_recordings`
        zurückgegeben), falls es für weitere Analysen benötigt wird.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = clean_recordings(
        recordings_dir, finger_idx=finger_idx, min_length=min_length, max_jump=max_jump
    )
    dataset = {label: trajs for label, trajs in dataset.items() if trajs}

    colors = _class_colors(sorted(dataset.keys()))

    _plot_trajectories(dataset, colors, output_dir)
    _plot_length_histogram(dataset, colors, output_dir)
    _plot_velocity_profiles(dataset, colors, output_dir)

    logger.info("Plots gespeichert unter: %s", output_dir)
    return dataset

def _accuracy(y_true, y_pred) -> float:
    """Anteil korrekter Vorhersagen (einfache Klassifikationsgenauigkeit)."""
    y_true = list(y_true)
    if not y_true:
        return 0.0
    correct = sum(true == pred for true, pred in zip(y_true, y_pred))
    return correct / len(y_true)


def _person_from_filename(path: str | Path) -> str | None:
    """
    Liest den Personen-Tag aus einem Aufnahme-Dateinamen.

    Aufnahmen heißen ``<L>-<person>-<n>.pkl`` (z.B. ``A-yannik-1.pkl``). Alte
    Aufnahmen ohne Person heißen ``<L>-<timestamp>.pkl`` — dort ist das zweite
    Feld eine Zahl. Für diese gibt es keinen Personen-Tag → ``None``.
    """
    parts = Path(path).stem.split("-")
    if len(parts) >= 3 and not parts[1].replace(".", "").isdigit():
        return parts[1]
    return None


def _load_by_person(
    recordings_dir: str | Path = "recordings",
    finger_idx: int = 8,
    min_length: int = 15,
    max_jump: float = 0.15,
) -> list[tuple[np.ndarray, str, str | None]]:
    """
    Lädt alle Aufnahmen und behält je Sequenz den Personen-Tag.

    Die Filter sind identisch zu :func:`dataset_building` (zu kurz /
    Tracking-Sprung / NaN werden verworfen), damit die Neue-Person-Bewertung
    dieselben Features nutzt wie das Standard-Training.

    Returns
    -------
    list of (traj, label, person)
        ``traj`` hat Form (N, 3) mit (x, y, velocity); ``person`` ist ``None``
        für alte Aufnahmen ohne Namen.
    """
    recordings_dir = Path(recordings_dir)
    samples: list[tuple[np.ndarray, str, str | None]] = []

    for label_dir in sorted(recordings_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for pkl_file in sorted(label_dir.glob("*.pkl")):
            with open(pkl_file, "rb") as f:
                recording = pickle.load(f)

            traj = _extract_trajectory(recording, finger_idx)
            if traj is None or len(traj) < min_length:
                continue
            if _is_outlier(traj, max_jump):
                continue
            traj = _to_features(traj)  # resample + normalize + velocity (wie dataset_building)
            if np.isnan(traj).any():  # gleicher NaN-Skip wie dataset_building
                continue

            samples.append((traj, label, _person_from_filename(pkl_file)))

    return samples


def _split_by_person(
    samples: list[tuple[np.ndarray, str, str | None]], held_out_person: str
) -> tuple[list, list]:
    """
    Teilt die Samples anhand der Person in Train/Test.

    Test = alle Sequenzen von ``held_out_person``; Train = alle übrigen
    (inklusive alter Aufnahmen ohne Namen). So testet man gegen eine Person,
    die das Modell nie gesehen hat — das simuliert den Prüfer.
    """
    train = [(traj, label) for traj, label, person in samples if person != held_out_person]
    test = [(traj, label) for traj, label, person in samples if person == held_out_person]
    return train, test


def _pack(pairs: list[tuple[np.ndarray, str]]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Baut aus (traj, label)-Paaren die (X, y, lengths)-Struktur für den Classifier."""
    seqs = [traj for traj, _ in pairs]
    labels = [label for _, label in pairs]
    X = np.vstack(seqs)
    y = np.array(labels)
    lengths = [len(seq) for seq in seqs]
    return X, y, lengths


def _plot_confusion_matrix(
    y_true, y_pred, classes, output_path: Path, title: str
) -> np.ndarray:
    """Zeichnet die Confusion Matrix als Heatmap und speichert sie als PNG."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 9))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation="vertical")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return cm


def evaluate_classifier(
    recordings_dir: str | Path = "recordings",
    output_dir: str | Path = "plots",
    held_out_person: str = "yannik",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Bewertet den HMM-Klassifikator mit zwei Zahlen und einer Confusion Matrix.

    Es werden absichtlich ZWEI Genauigkeiten berechnet, weil sie zwei
    verschiedene Fragen beantworten:

    1. **Standard-Accuracy** (gleiche Personen, unbekannte Aufnahmen): Der
       Datensatz wird über :func:`dataset_building` sequenzweise & stratifiziert
       in Train/Test getrennt (kein Data-Leakage). Darauf wird ein
       :class:`HMMClassifier` trainiert und Accuracy + eine Confusion Matrix auf
       dem Test-Split berechnet. Diese Zahl ist optimistisch, weil jede Person
       auch im Training vorkommt.
    2. **Neue-Person-Accuracy** (Generalisierung): Das gleiche Modell wird einmal
       komplett OHNE ``held_out_person`` trainiert und nur auf dieser Person
       getestet. Das simuliert den Prüfer, den das Modell nie gesehen hat, und
       ist die ehrliche Schätzung der Generalisierung.

    Der Klassifikator wird hier selbst trainiert (statt ``data/hmm.pkl`` zu
    laden), damit Modell und Test-Split garantiert aus demselben Split stammen —
    das schließt versehentliches Data-Leakage aus.

    Parameters
    ----------
    recordings_dir : str or Path
        Verzeichnis mit den Aufnahmen (``recordings/<label>/*.pkl``).
    output_dir : str or Path
        Zielverzeichnis für die Confusion-Matrix-PNG.
    held_out_person : str
        Person, die für die Neue-Person-Bewertung komplett zurückgehalten wird.
        Sollte alle Klassen abdecken (Default: ``"yannik"``).
    test_size, random_state
        Parameter für den stratifizierten Standard-Split.

    Returns
    -------
    dict
        ``{"accuracy_standard", "accuracy_new_person", "held_out_person"}``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Teil A: Standard-Bewertung (gleiche Personen, unbekannte Aufnahmen) ---
    data = dataset_building(
        "data/dataset.pkl",
        recordings_dir=recordings_dir,
        test_size=test_size,
        random_state=random_state,
    )
    clf = HMMClassifier().fit(data["X_train"], data["y_train"], data["lengths_train"])
    y_pred = clf.predict(data["X_test"], data["lengths_test"])
    acc_standard = _accuracy(data["y_test"], y_pred)
    _plot_confusion_matrix(
        data["y_test"],
        y_pred,
        data["classes"],
        output_dir / "confusion_matrix.png",
        f"Confusion Matrix — Standard-Test (Accuracy {acc_standard:.1%})",
    )

    # --- Teil B: Neue-Person-Bewertung (eine Person komplett raushalten) ---
    samples = _load_by_person(recordings_dir)
    train_pairs, test_pairs = _split_by_person(samples, held_out_person)
    if not test_pairs:
        raise ValueError(
            f"Keine Aufnahmen fuer held_out_person='{held_out_person}' gefunden. "
            "Bitte eine Person angeben, die als <L>-<person>-<n>.pkl aufgenommen hat."
        )
    if not train_pairs:
        raise ValueError("Kein Trainingsmaterial nach dem Personen-Split uebrig.")

    X_tr, y_tr, len_tr = _pack(train_pairs)
    X_te, y_te, len_te = _pack(test_pairs)

    missing = sorted(set(y_te.tolist()) - set(y_tr.tolist()))
    if missing:
        logger.warning(
            "Klassen ohne Trainingsdaten nach Holdout (koennen nie erkannt werden): %s",
            missing,
        )

    clf_holdout = HMMClassifier().fit(X_tr, y_tr, len_tr)
    acc_new = _accuracy(y_te, clf_holdout.predict(X_te, len_te))

    logger.info(
        "Standard (gleiche Personen): %.1f%%  |  Neue Person (%s): %.1f%%",
        acc_standard * 100,
        held_out_person,
        acc_new * 100,
    )

    return {
        "accuracy_standard": acc_standard,
        "accuracy_new_person": acc_new,
        "held_out_person": held_out_person,
    }


# --- Replay / Galerie der Rohaufnahmen -------------------------------------
#
# Jede einzelne Aufnahme bekommt einen Status. "ok" ist eine saubere Aufnahme,
# alles andere ist ein Grund, sie kritisch anzuschauen. Wir werfen schlechte
# Aufnahmen hier NICHT weg (anders als clean_recordings) -- die Galerie soll sie
# ja gerade sichtbar machen.
_STATUS_OK = "ok"
_STATUS_KURZ = "kurz"              # zu wenige Frames (unter min_length)
_STATUS_SPRUNG = "sprung"          # Tracking-Sprung: Hand kurz verloren und weit weg wiedergefunden
_STATUS_KEINE_HAND = "keine_hand"  # in keinem Frame eine Hand erkannt

# Kurze rote Beschriftung fuer markierte Kacheln.
_STATUS_LABELS = {
    _STATUS_KURZ: "kurz",
    _STATUS_SPRUNG: "Sprung",
    _STATUS_KEINE_HAND: "keine Hand",
}


def _load_recordings_with_status(
    recordings_dir: str | Path,
    finger_idx: int = 8,
    min_length: int = 15,
    max_jump: float = 0.15,
) -> dict[str, list[dict]]:
    """
    Laedt JEDE Aufnahme (auch die schlechten) und haengt einen Status an.

    Die Filtergrenze ist dieselbe wie in :func:`clean_recordings` (dem
    Training) -- nur dass wir hier nichts wegwerfen, sondern jede Aufnahme
    markieren. Zusaetzlich unterscheiden wir "gar keine Hand erkannt" extra
    (das faellt beim Training unter "zu kurz"). So sieht man auf einen Blick,
    welche Aufnahmen ins Training kommen und welche aussortiert werden.

    Parameters
    ----------
    recordings_dir : str or Path
        Verzeichnis mit den Label-Unterordnern (``recordings/<label>/*.pkl``).
    finger_idx : int
        MediaPipe-Landmark-Index des verfolgten Fingers (8 = Zeigefingerspitze).
    min_length, max_jump
        Gleiche Schwellen wie beim Training (siehe :func:`clean_recordings`).

    Returns
    -------
    dict[str, list[dict]]
        Pro Buchstabe eine Liste von Eintraegen
        ``{"name": str, "traj": np.ndarray | None, "status": str}``.
        ``traj`` ist die normalisierte (x, y)-Bahn zum Anzeigen, oder ``None``,
        wenn gar keine Hand erkannt wurde.
    """
    recordings_dir = Path(recordings_dir)
    galerie: dict[str, list[dict]] = {}

    for label_dir in sorted(recordings_dir.iterdir()):
        if not label_dir.is_dir():
            continue

        eintraege: list[dict] = []
        for pkl_file in sorted(label_dir.glob("*.pkl")):
            with open(pkl_file, "rb") as f:
                recording = pickle.load(f)

            traj = _extract_trajectory(recording, finger_idx)

            # Status nach denselben Regeln wie clean_recordings bestimmen.
            # Reihenfolge ist wichtig: erst "gar keine Hand", dann "zu kurz",
            # dann "Sprung", sonst "ok".
            if traj is None:
                status = _STATUS_KEINE_HAND
            elif len(traj) < min_length:
                status = _STATUS_KURZ
            elif _is_outlier(traj, max_jump):
                status = _STATUS_SPRUNG
            else:
                status = _STATUS_OK

            # Fuer die Anzeige Lage und Groesse angleichen, damit alle Kacheln
            # vergleichbar aussehen (egal wo/wie gross gemalt wurde).
            traj_xy = _normalize(traj) if traj is not None else None
            eintraege.append({"name": pkl_file.stem, "traj": traj_xy, "status": status})

        galerie[label_dir.name] = eintraege

    return galerie


def _plot_letter_gallery(
    label: str, eintraege: list[dict], color, output_path: Path, cols: int
) -> None:
    """
    Zeichnet alle Aufnahmen EINES Buchstabens als Kachel-Raster in ein PNG.

    Jede Kachel zeigt die (x, y)-Bahn einer einzelnen Aufnahme mit Startpunkt.
    Markierte (schlechte) Aufnahmen bekommen einen roten Rahmen und den Grund
    als rote Beschriftung -- so springt jede problematische Sequenz ins Auge.
    """
    n = len(eintraege)
    # Bei wenigen Aufnahmen nicht unnoetig breit werden: hoechstens n Spalten.
    cols = min(cols, max(n, 1))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.4 * cols, 1.6 * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, eintrag in zip(axes, eintraege):
        traj = eintrag["traj"]
        status = eintrag["status"]

        # Die Bahn zeichnen (falls eine Hand da war).
        if traj is not None and len(traj) > 0:
            ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1)
            # Startpunkt markieren -- aber nur, wenn der erste Punkt gueltig ist.
            if not np.isnan(traj[0]).any():
                ax.plot(traj[0, 0], traj[0, 1], "o", color=color, markersize=3)

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        # Bild-/MediaPipe-y waechst nach UNTEN, matplotlib-y nach OBEN. Ohne diese
        # Zeile stuenden die Buchstaben auf dem Kopf. invert_yaxis() bringt die
        # Bahn in echte Bildkoordinaten, also aufrecht wie tatsaechlich gezeichnet.
        ax.invert_yaxis()

        # Schlechte Aufnahmen rot umranden und den Grund darunter schreiben.
        if status != _STATUS_OK:
            for seite in ax.spines.values():
                seite.set_color("red")
                seite.set_linewidth(2)
            ax.set_xlabel(_STATUS_LABELS[status], color="red", fontsize=7)

    # Uebrige leere Kacheln am Ende ausblenden.
    for ax in axes[n:]:
        ax.axis("off")

    markiert = sum(1 for e in eintraege if e["status"] != _STATUS_OK)
    fig.suptitle(f"{label} — {n} Aufnahmen ({markiert} markiert)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def replay_recordings(
    recordings_dir: str | Path = "recordings",
    output_dir: str | Path = "plots/gallery",
    cols: int = 10,
    finger_idx: int = 8,
    min_length: int = 15,
    max_jump: float = 0.15,
) -> dict[str, dict]:
    """
    Statisches "Replay" der Rohaufnahmen als Kachel-Galerie.

    Erfuellt das Kriterium *"Exploration und Replay der aufgenommenen
    Rohdaten"*: Statt die Aufnahmen als Video abzuspielen, legen wir sie als
    Bild-Galerie nebeneinander -- das ist reproduzierbar, teilbar (PNG) und
    zeigt alle Beispiele gleichzeitig.

    Pro Buchstabe wird EIN PNG unter ``output_dir`` erzeugt
    (``gallery_<Buchstabe>.png``). Jede einzelne Aufnahme ist eine Kachel mit
    ihrer (x, y)-Bahn. Schlechte Aufnahmen (zu kurz, Tracking-Sprung, keine
    Hand) sind rot umrandet samt Grund -- so sieht man auf einen Blick, welche
    Beispiele gut und welche problematisch sind. Damit sind gleich beide
    optionalen Erweiterungen abgedeckt: *automatisches Filtern schlechter
    Sequenzen* (Markierung) und *Kombination mit Visualisierung*.

    Parameters
    ----------
    recordings_dir : str or Path
        Verzeichnis mit den Rohaufnahmen (``recordings/<label>/*.pkl``).
    output_dir : str or Path
        Zielverzeichnis fuer die PNG-Galerien (wird angelegt, falls noetig).
    cols : int
        Anzahl Kacheln pro Zeile.
    finger_idx, min_length, max_jump
        Gleiche Parameter/Schwellen wie beim Training.

    Returns
    -------
    dict[str, dict]
        Pro Buchstabe eine Zusammenfassung
        ``{"gesamt", "ok", "markiert", "pfad"}`` -- praktisch fuer eine
        schnelle Qualitaets-Uebersicht im Terminal.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    galerie = _load_recordings_with_status(
        recordings_dir, finger_idx=finger_idx, min_length=min_length, max_jump=max_jump
    )
    colors = _class_colors(sorted(galerie.keys()))

    zusammenfassung: dict[str, dict] = {}
    for label in sorted(galerie.keys()):
        eintraege = galerie[label]
        if not eintraege:
            continue

        pfad = output_dir / f"gallery_{label}.png"
        _plot_letter_gallery(label, eintraege, colors[label], pfad, cols)

        markiert = sum(1 for e in eintraege if e["status"] != _STATUS_OK)
        zusammenfassung[label] = {
            "gesamt": len(eintraege),
            "ok": len(eintraege) - markiert,
            "markiert": markiert,
            "pfad": str(pfad),
        }
        logger.info(
            "[%s] %d Aufnahmen, %d markiert -> %s",
            label,
            len(eintraege),
            markiert,
            pfad,
        )

    logger.info("Galerie gespeichert unter: %s", output_dir)
    return zusammenfassung
