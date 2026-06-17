import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from GestureRecognition.labeling import clean_recordings

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

def evaluate_classifier():
    """
    TODO: Evaluation deines Klassifikators

    Ziel:
    -----
    Implementiere eine sinnvolle Auswertung deines Modells auf Testdaten.

    Warum ist das wichtig?
    ----------------------
    - Du brauchst objektive Metriken für die Qualität deines Modells
    - Training allein reicht nicht, entscheidend ist die Generalisierung

    Anforderungen / Ideen:
    ----------------------
    - Lade ein trainiertes Modell
    - Lade Testdaten (getrennt vom Training!)
    - Berechne Vorhersagen
    - Vergleiche Vorhersagen mit Ground Truth

    Metriken:
    ---------
    - Klassifikationsgenauigkeit (Accuracy)
    - Confusion Matrix

    .. tip::
       Eine Confusion Matrix zeigt dir:
         - Welche Klassen gut erkannt werden
         - Wo dein Modell Fehler macht

    .. warning::
       Testdaten dürfen **nicht** aus dem Training stammen!

    Interpretation:
    ---------------
    Du solltest erklären können:
    - Welche Klassen gut funktionieren
    - Welche Klassen verwechselt werden
    - Warum das passieren könnte

    .. note::
       Schlechte Performance liegt oft an:
         - schlechten Trainingsdaten
         - zu wenigen Beispielen
         - ungeeigneten Features

    Erweiterung (optional):
    -----------------------
    - Weitere Metriken (Precision, Recall, F1)
    - Vergleich verschiedener Modelle
    """
    pass


def replay_recordings():
    """
    TODO: Exploration und Replay der aufgenommenen Rohdaten

    Ziel:
    -----
    Ermögliche es, aufgenommene Sequenzen erneut abzuspielen
    und qualitativ zu überprüfen.

    Warum ist das wichtig?
    ----------------------
    - Du kannst überprüfen, ob deine Aufnahmen korrekt sind
    - Fehler in der Datenerfassung werden früh sichtbar
    - Du entwickelst ein besseres Verständnis für deine Daten

    Anforderungen / Ideen:
    ----------------------
    - Lade gespeicherte Aufnahmen
    - Spiele diese erneut ab (z. B. über SignalHub / Replay-Modus)
    - Iteriere über verschiedene Labels und Beispiele

    .. tip::
       Besonders hilfreich:
         - Vergleiche mehrere Beispiele derselben Klasse
         - Suche nach inkonsistenten Bewegungen

    .. warning::
       Schlechte oder inkonsistente Aufnahmen führen fast immer zu
       schlechten Modellen. Überprüfe deine Daten frühzeitig!

    Abgabe:
    -------
    - Du solltest zeigen können, wie deine Daten aussehen (Replay)
    - Du solltest erklären können:
        - Welche Beispiele gut sind
        - Welche problematisch sind

    Erweiterung (optional):
    -----------------------
    - Automatisches Filtern schlechter Sequenzen
    - Kombination mit Visualisierung
    """
    pass