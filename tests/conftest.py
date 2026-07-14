"""Gemeinsame Test-Fixtures.

Zentraler Zweck (Issue #62): die modellabhängigen End-to-End-Tests dürfen in
einem frischen Clone NICHT stillschweigend übersprungen werden. Das trainierte
Modell liegt unter ``data/hmm.pkl``, aber ``data/`` ist per ``.gitignore``
ausgeschlossen -- in einem sauberen Clone fehlt es also. Statt via ``skipif`` zu
überspringen wird es hier EINMAL pro Testlauf deterministisch aus ``recordings/``
trainiert.

Zusätzlich (Issue #61): eine zweite Fixture trainiert ein Hold-out-Modell, in
dem die Testperson ``HOLDOUT_PERSON`` nachweislich NICHT vorkommt -- damit der
End-to-End-Test mit einer wirklich unbekannten Person laufen kann. Aufnahmen
ohne Personen-Tag (alte Timestamp-Dateien) werden dabei ebenfalls
ausgeschlossen, weil sich bei ihnen nicht beweisen lässt, von wem sie stammen.
"""
import json
import logging
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "data" / "hmm.pkl"

# Person, die für den Unbekannte-Person-Test komplett aus dem Training fliegt.
# wayan hat alle 26 Buchstaben und laut docs/ergebnisse.md den besten Hold-out
# (80% best_label) -- eine stabile Wahl. Bewusst NICHT yannik: der ist schon die
# Testperson des Bekannte-Person-Tests (test_end_to_end.py).
HOLDOUT_PERSON = "wayan"
HOLDOUT_MODEL_PATH = ROOT / "data" / f"hmm_ohne_{HOLDOUT_PERSON}.pkl"
# Sidecar-Datei: hält fest, mit welchen Aufnahmen das Hold-out-Modell trainiert
# wurde. Ändert sich der Aufnahmebestand, passt die Liste nicht mehr und das
# Modell wird automatisch neu trainiert (kein stiller Betrug durch Cache).
HOLDOUT_MANIFEST_PATH = ROOT / "data" / f"hmm_ohne_{HOLDOUT_PERSON}.trainfiles.json"

_log = logging.getLogger("tests.model")


def prepare_hmm_model() -> Path:
    """Sorgt dafür, dass ``data/hmm.pkl`` existiert und meldet klar, ob geladen
    oder trainiert wurde. Ausgelagert, damit sowohl die pytest-Fixture als auch der
    Standalone-Aufruf (``python tests/test_end_to_end.py``) dieselbe Logik nutzen.
    """
    if MODEL_PATH.exists():
        _log.warning("HMM-Modell vorhanden -> geladen: %s", MODEL_PATH)
        return MODEL_PATH

    _log.warning("HMM-Modell fehlt -> trainiere deterministisch aus recordings/ ...")
    from train import train

    start = time.perf_counter()
    train(model_path=str(MODEL_PATH))
    _log.warning(
        "HMM-Modell in %.1f s trainiert -> %s", time.perf_counter() - start, MODEL_PATH
    )
    return MODEL_PATH


@pytest.fixture(scope="session")
def ensure_hmm_model() -> Path:
    """Garantiert vor den modellabhängigen Tests, dass ``data/hmm.pkl`` da ist --
    trainiert es bei Bedarf. Kein ``skipif`` mehr: fehlt das Modell im frischen
    Clone, wird es trainiert; ist es kaputt/leer, schlagen die Tests fehl (rot),
    statt still übersprungen zu werden.
    """
    return prepare_hmm_model()


def _holdout_training_files() -> list[str]:
    """Bestimmt die Trainingsdateien für das Hold-out-Modell (Issue #61).

    Regel: Es kommen NUR Aufnahmen mit bekanntem Personen-Tag hinein, deren
    Person nicht ``HOLDOUT_PERSON`` ist. Dateien ohne Tag fliegen raus, weil
    nicht beweisbar ist, wer sie aufgenommen hat -- genau darum geht es bei
    "wirklich unbekannte Person" ja.
    """
    from GestureRecognition.visualization import _person_from_filename

    files = []
    for pkl_file in sorted((ROOT / "recordings").glob("*/*.pkl")):
        person = _person_from_filename(pkl_file)
        if person is not None and person != HOLDOUT_PERSON:
            files.append(str(pkl_file.relative_to(ROOT)))
    return files


def prepare_holdout_model() -> Path:
    """Sorgt dafür, dass das Modell OHNE ``HOLDOUT_PERSON`` existiert.

    Trainiert deterministisch (random_state=42, n_components wie train.py) auf
    allen Aufnahmen mit bekanntem Personen-Tag ausser der Hold-out-Person.
    Ein bereits vorhandenes Modell wird nur wiederverwendet, wenn die im
    Manifest festgehaltene Trainingsdatei-Liste noch exakt zum aktuellen
    Aufnahmebestand passt -- sonst wird neu trainiert.
    """
    train_files = _holdout_training_files()

    # Beweis-Check (Akzeptanzkriterium): keine einzige Trainingsdatei darf von
    # der Hold-out-Person stammen. Nach der Filterregel oben ist das immer so;
    # der explizite Check schützt gegen künftige Änderungen an der Regel.
    from GestureRecognition.visualization import _person_from_filename

    verraeter = [f for f in train_files if _person_from_filename(ROOT / f) == HOLDOUT_PERSON]
    assert not verraeter, f"Hold-out verletzt: {verraeter[:5]}"

    if HOLDOUT_MODEL_PATH.exists() and HOLDOUT_MANIFEST_PATH.exists():
        manifest = json.loads(HOLDOUT_MANIFEST_PATH.read_text(encoding="utf-8"))
        if manifest.get("train_files") == train_files:
            _log.warning(
                "Hold-out-Modell (ohne %s) vorhanden -> geladen: %s",
                HOLDOUT_PERSON, HOLDOUT_MODEL_PATH,
            )
            return HOLDOUT_MODEL_PATH
        _log.warning("Aufnahmebestand hat sich geändert -> Hold-out-Modell wird neu trainiert.")

    _log.warning(
        "Trainiere Hold-out-Modell ohne '%s' (%d Trainingsdateien, nur mit Personen-Tag) ...",
        HOLDOUT_PERSON, len(train_files),
    )
    from GestureRecognition.hmmclassifier import HMMClassifier
    from GestureRecognition.visualization import _load_by_person, _pack

    start = time.perf_counter()
    samples = _load_by_person(ROOT / "recordings")
    train_pairs = [
        (traj, label)
        for traj, label, person in samples
        if person is not None and person != HOLDOUT_PERSON
    ]
    X, y, lengths = _pack(train_pairs)
    clf = HMMClassifier().fit(X, y, lengths)
    clf.save(HOLDOUT_MODEL_PATH)
    HOLDOUT_MANIFEST_PATH.write_text(
        json.dumps(
            {"holdout_person": HOLDOUT_PERSON, "train_files": train_files},
            ensure_ascii=False, indent=1,
        ),
        encoding="utf-8",
    )
    _log.warning(
        "Hold-out-Modell in %.1f s trainiert (%d Sequenzen, %d Klassen) -> %s",
        time.perf_counter() - start, len(train_pairs), len(clf.classes_), HOLDOUT_MODEL_PATH,
    )
    return HOLDOUT_MODEL_PATH


@pytest.fixture(scope="session")
def ensure_holdout_model() -> Path:
    """Garantiert das Hold-out-Modell (ohne ``HOLDOUT_PERSON``) für den
    Unbekannte-Person-End-to-End-Test -- trainiert es bei Bedarf.
    """
    return prepare_holdout_model()
