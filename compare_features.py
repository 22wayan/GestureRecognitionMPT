"""
Vergleicht die zwei Feature-Formate aus Issue #59 reproduzierbar:

  Baseline    "xyv"      -> (x, y, speed)
  Alternative "xydxdyv"  -> (x, y, dx, dy, speed)   (zusaetzlich Bewegungsrichtung)

Fuer beide Varianten werden mit DEMSELBEN Split berechnet:

  1. Standard-Accuracy (sequenzweiser, stratifizierter 80/20-Split), und
  2. der Personen-Hold-out fuer JEDE benannte Person: Modell ohne diese Person
     trainieren, nur auf ihr testen. Berichtet werden Mittelwert und Minimum --
     das Minimum zeigt den schlechtesten Fall (die "ehrliche" Generalisierung).

Das Ergebnis wird als Markdown-Tabelle nach reports/feature_vergleich.md
geschrieben. Gewechselt wird nur, wenn die Generalisierung (Hold-out) messbar
besser oder mindestens stabiler wird -- sonst bleibt die Baseline.

Benutzung:
    python compare_features.py
"""

import logging
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from GestureRecognition.hmmclassifier import HMMClassifier
from GestureRecognition.labeling import (
    _extract_trajectory,
    _is_outlier,
    _to_features,
)
from GestureRecognition.visualization import _person_from_filename

logger = logging.getLogger(__name__)

FEATURE_SETS = ["xyv", "xydxdyv"]
RECORDINGS_DIR = Path("recordings")
REPORT_PATH = Path("reports/feature_vergleich.md")
MIN_LENGTH = 15
MAX_JUMP = 0.15
TEST_SIZE = 0.2
RANDOM_STATE = 42


def _load_raw_samples() -> list[tuple[np.ndarray, str, str | None]]:
    """Laedt alle Aufnahmen EINMAL als Roh-Trajektorien (vor der Feature-Wahl).

    Filter identisch zu dataset_building: zu kurz / Tracking-Sprung raus.
    Rueckgabe: Liste von (roh_trajektorie, label, person).
    """
    samples = []
    for label_dir in sorted(RECORDINGS_DIR.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for pkl_file in sorted(label_dir.glob("*.pkl")):
            with open(pkl_file, "rb") as f:
                recording = pickle.load(f)
            traj = _extract_trajectory(recording, finger_idx=8)
            if traj is None or len(traj) < MIN_LENGTH:
                continue
            if _is_outlier(traj, MAX_JUMP):
                continue
            samples.append((traj, label, _person_from_filename(pkl_file)))
    return samples


def _featurize(raw_samples, feature_set):
    """Wendet die zentrale Feature-Funktion auf alle Roh-Samples an (NaN-Skip
    wie in dataset_building)."""
    out = []
    for traj, label, person in raw_samples:
        feats = _to_features(traj, feature_set=feature_set)
        if np.isnan(feats).any():
            continue
        out.append((feats, label, person))
    return out


def _pack(pairs):
    """Baut aus (traj, label)-Paaren die (X, y, lengths)-Struktur fuer den Classifier."""
    seqs = [traj for traj, _ in pairs]
    X = np.vstack(seqs)
    y = np.array([label for _, label in pairs])
    lengths = [len(seq) for seq in seqs]
    return X, y, lengths


def _accuracy(y_true, y_pred) -> float:
    """Anteil korrekter Vorhersagen (einfache Klassifikationsgenauigkeit)."""
    y_true = list(y_true)
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def _standard_accuracy(samples) -> float:
    """Sequenzweiser, stratifizierter 80/20-Split -- wie dataset_building."""
    pairs = [(traj, label) for traj, label, _ in samples]
    labels = [label for _, label in pairs]
    indices = np.arange(len(pairs))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE
    )
    X_tr, y_tr, len_tr = _pack([pairs[i] for i in train_idx])
    X_te, y_te, len_te = _pack([pairs[i] for i in test_idx])
    clf = HMMClassifier(random_state=RANDOM_STATE).fit(X_tr, y_tr, len_tr)
    return _accuracy(y_te, clf.predict(X_te, len_te))


def _holdout_accuracies(samples) -> dict[str, float]:
    """Personen-Hold-out fuer jede benannte Person: ohne sie trainieren, nur
    auf ihr testen."""
    persons = sorted({p for _, _, p in samples if p is not None})
    result = {}
    for person in persons:
        train_pairs = [(t, l) for t, l, p in samples if p != person]
        test_pairs = [(t, l) for t, l, p in samples if p == person]
        if not test_pairs or not train_pairs:
            continue
        X_tr, y_tr, len_tr = _pack(train_pairs)
        X_te, y_te, len_te = _pack(test_pairs)
        clf = HMMClassifier(random_state=RANDOM_STATE).fit(X_tr, y_tr, len_tr)
        result[person] = _accuracy(y_te, clf.predict(X_te, len_te))
    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("GestureRecognition").setLevel(logging.WARNING)

    logger.info("Lade Aufnahmen aus '%s' ...", RECORDINGS_DIR)
    raw = _load_raw_samples()
    logger.info("%d gueltige Aufnahmen geladen.", len(raw))

    results = {}
    for feature_set in FEATURE_SETS:
        logger.info("== Feature-Set '%s' ==", feature_set)
        samples = _featurize(raw, feature_set)
        std = _standard_accuracy(samples)
        logger.info("Standard-Accuracy: %.3f", std)
        holdout = _holdout_accuracies(samples)
        for person, acc in holdout.items():
            logger.info("Hold-out %-8s: %.3f", person, acc)
        values = list(holdout.values())
        results[feature_set] = {
            "standard": std,
            "holdout": holdout,
            "holdout_mean": float(np.mean(values)) if values else float("nan"),
            "holdout_min": float(np.min(values)) if values else float("nan"),
        }

    # Entscheidung: Wechsel nur bei messbar besserer ODER stabilerer
    # Generalisierung (Mittelwert und Minimum des Hold-outs nicht schlechter,
    # mindestens eines besser).
    base, alt = results["xyv"], results["xydxdyv"]
    better = (
        alt["holdout_mean"] >= base["holdout_mean"]
        and alt["holdout_min"] >= base["holdout_min"]
        and (
            alt["holdout_mean"] > base["holdout_mean"]
            or alt["holdout_min"] > base["holdout_min"]
        )
    )
    empfehlung = "xydxdyv" if better else "xyv"

    persons = sorted(
        set(base["holdout"]) | set(alt["holdout"])
    )
    lines = [
        "# Feature-Vergleich: (x, y, speed) vs. (x, y, dx, dy, speed)",
        "",
        "Issue #59 -- reproduzierbar mit `python compare_features.py`",
        f"(gleicher Split fuer beide Varianten, random_state={RANDOM_STATE}).",
        "",
        "| Metrik | xyv (Baseline) | xydxdyv (mit Richtung) |",
        "| --- | --- | --- |",
        f"| Standard-Accuracy | {base['standard']:.3f} | {alt['standard']:.3f} |",
    ]
    for person in persons:
        b = base["holdout"].get(person)
        a = alt["holdout"].get(person)
        lines.append(
            f"| Hold-out {person} | "
            f"{'-' if b is None else f'{b:.3f}'} | "
            f"{'-' if a is None else f'{a:.3f}'} |"
        )
    lines += [
        f"| **Hold-out Mittelwert** | **{base['holdout_mean']:.3f}** | **{alt['holdout_mean']:.3f}** |",
        f"| **Hold-out Minimum** | **{base['holdout_min']:.3f}** | **{alt['holdout_min']:.3f}** |",
        "",
        "## Entscheidung",
        "",
        f"Gewaehlt: **{empfehlung}** (`FEATURE_SET` in `GestureRecognition/labeling.py`).",
        "",
        "Regel: Gewechselt wird nur, wenn Mittelwert und Minimum des",
        "Personen-Hold-outs nicht schlechter und mindestens eines besser wird.",
        "Der Hold-out zaehlt mehr als die Standard-Accuracy, weil er die",
        "Generalisierung auf eine unbekannte Person misst (Pruefer-Szenario).",
        "",
        "## Feature-Bedeutung in einfachen Worten",
        "",
        "- `x`, `y`: Wo ist der Finger (nach Zentrieren + Skalieren)?",
        "- `speed`: Wie stark bewegt er sich gerade (Laenge des Schritts)?",
        "- `dx`, `dy` (nur Alternative): In welche RICHTUNG geht der Schritt?",
        "  Zwei Buchstaben mit aehnlicher Form, aber anderer Zeichenrichtung",
        "  erzeugen dieselbe speed-Spur, aber unterschiedliche dx/dy-Spuren.",
        "  Preis: 5 statt 3 Features -> mehr Parameter pro HMM-Zustand ->",
        "  mehr Overfitting-Risiko bei wenig Daten.",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report geschrieben: %s", REPORT_PATH)
    logger.info("Empfehlung: FEATURE_SET = '%s'", empfehlung)


if __name__ == "__main__":
    main()
