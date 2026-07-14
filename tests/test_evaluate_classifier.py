"""
Kleiner Selbstcheck für die Hilfslogik von ``evaluate_classifier``.

Bewusst ohne pytest-Setup — direkt lauffähig:

    python tests/test_evaluate_classifier.py

Getestet wird nur die einzige knifflige Logik (Personen-Parsing, Personen-Split,
Accuracy). Die Plot-/Modell-Integration wird per End-to-End-Lauf verifiziert.
"""

import sys
from pathlib import Path

import numpy as np

# Projekt-Root auf den Importpfad legen, damit das Paket ohne Installation läuft.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from GestureRecognition.visualization import (  # noqa: E402
    _accuracy,
    _person_from_filename,
    _split_by_person,
)


def test_person_from_filename():
    assert _person_from_filename("recordings/A/A-yannik-1.pkl") == "yannik"
    assert _person_from_filename("A-wayan-15.pkl") == "wayan"
    assert _person_from_filename("Z-arian-3.pkl") == "arian"
    # Alte Aufnahmen ohne Personen-Tag (Timestamp) → None
    assert _person_from_filename("A-1773050612.172112.pkl") is None


def test_accuracy():
    assert _accuracy(["A", "B", "C"], ["A", "B", "C"]) == 1.0
    assert _accuracy(["A", "B"], ["A", "X"]) == 0.5
    assert _accuracy([], []) == 0.0


def test_split_by_person():
    dummy = np.zeros((3, 3))
    samples = [
        (dummy, "A", "yannik"),
        (dummy, "B", "yannik"),
        (dummy, "A", "wayan"),
        (dummy, "C", None),  # unbekannte Herkunft → aus Hold-out ausschliessen
    ]
    train, test = _split_by_person(samples, "yannik")

    # Die Holdout-Person ist KOMPLETT im Test und GAR NICHT im Training.
    assert len(test) == 2
    assert len(train) == 1
    test_labels = sorted(label for _, label in test)
    train_labels = sorted(label for _, label in train)
    assert test_labels == ["A", "B"]
    assert train_labels == ["A"]


if __name__ == "__main__":
    test_person_from_filename()
    test_accuracy()
    test_split_by_person()
    print("OK: alle Selbstchecks bestanden")
