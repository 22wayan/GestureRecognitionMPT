"""Gemeinsame Test-Fixtures.

Zentraler Zweck (Issue #62): die modellabhängigen End-to-End-Tests dürfen in
einem frischen Clone NICHT stillschweigend übersprungen werden. Das trainierte
Modell liegt unter ``data/hmm.pkl``, aber ``data/`` ist per ``.gitignore``
ausgeschlossen -- in einem sauberen Clone fehlt es also. Statt via ``skipif`` zu
überspringen wird es hier EINMAL pro Testlauf deterministisch aus ``recordings/``
trainiert.
"""
import logging
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "data" / "hmm.pkl"
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
