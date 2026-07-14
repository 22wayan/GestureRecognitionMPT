"""
Regressionstest: der HMMClassifier darf keine Klasse mit einem NaN-Modell
hinterlassen (stiller 0%-Recall).

Hintergrund
-----------
Ein GaussianHMM kann bei zu wenigen Trainingssequenzen für die gewählte Anzahl
Zustände in NaN kippen: ein Zustand bekommt kaum Daten, seine Varianz kollabiert
trotz ``min_covar``, und ``score()`` liefert für diese Klasse ``NaN``/``-inf`` ->
die Klasse wird NIE vorhergesagt. Genau das ist im Live-Test mit einer neu
aufgenommenen Geste ("Dreieck", 8 Trainingssequenzen) bei ``n_components=10``
passiert: 0/10 erkannt, obwohl das Modell "trainiert" wurde.

``HMMClassifier.fit`` fängt das jetzt ab, indem es die Zustandszahl für eine
kollabierende Klasse schrittweise reduziert, bis das Modell stabil ist. Dieser
Test hält den Schutz fest.

Laeuft unter pytest und standalone:

    python tests/test_hmm_nan_guard.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from GestureRecognition.hmmclassifier import HMMClassifier  # noqa: E402


def _kollaps_sequenzen() -> list[np.ndarray]:
    """Wenige, fast identische kurze Sequenzen im (x, y, velocity)-Format.

    Deterministisch (fester Seed). Bei ``n_components=10`` ist ein rohes
    GaussianHMM auf diesen Daten überparametrisiert und kippt in NaN -- genau der
    Fall, den der Guard abfangen muss.
    """
    rng = np.random.RandomState(0)
    base = np.linspace(0.0, 1.0, 20)
    template = np.stack([base, base ** 2, np.gradient(base)], axis=1)
    return [template + rng.normal(0.0, 1e-4, template.shape) for _ in range(3)]


def test_fit_liefert_nie_ein_nan_modell():
    """KernInvariante: nach ``fit`` darf kein Klassenmodell NaN-Parameter haben,
    selbst wenn ``n_components`` für die Datenmenge zu groß ist. Ohne den Guard
    würde ``means_`` hier NaN -> die Klasse wäre nie vorhersagbar.
    """
    sequences = _kollaps_sequenzen()
    X = np.vstack(sequences)
    lengths = [len(s) for s in sequences]
    y = ["GESTE"] * len(sequences)

    clf = HMMClassifier(n_components=10, random_state=42)
    clf.fit(X, y, lengths)

    model = clf._models["GESTE"]
    assert not np.isnan(model.means_).any(), "means_ enthält NaN -- Guard hat nicht gegriffen."
    assert not np.isnan(model.covars_).any(), "covars_ enthält NaN -- Guard hat nicht gegriffen."
    # Der Guard darf die Zustandszahl nur reduzieren, nie erhöhen.
    assert 1 <= model.n_components <= 10

    # Und die Klasse muss vorhersagbar bleiben (kein reiner -inf-Score).
    assert clf.predict(X, lengths) == ["GESTE"] * len(sequences)


def test_gut_besetzte_klasse_behaelt_volle_zustandszahl():
    """Gegenprobe: bei genug, hinreichend variablen Sequenzen bleibt die volle
    Zustandszahl erhalten -- der Guard greift nur im Notfall.
    """
    rng = np.random.RandomState(1)
    base = np.linspace(0.0, 1.0, 48)
    sequences = [
        np.stack([base, np.sin(base * (3 + i)), np.gradient(base)], axis=1)
        + rng.normal(0.0, 0.05, (48, 3))
        for i in range(30)
    ]
    X = np.vstack(sequences)
    lengths = [len(s) for s in sequences]
    y = ["GESTE"] * len(sequences)

    clf = HMMClassifier(n_components=5, random_state=42)
    clf.fit(X, y, lengths)

    model = clf._models["GESTE"]
    assert not np.isnan(model.means_).any()
    assert model.n_components == 5, "Guard hat unnötig reduziert."


if __name__ == "__main__":
    test_fit_liefert_nie_ein_nan_modell()
    test_gut_besetzte_klasse_behaelt_volle_zustandszahl()
    print("OK: HMMClassifier faengt NaN-Kollaps ab, ohne gut besetzte Klassen zu beschneiden.")
