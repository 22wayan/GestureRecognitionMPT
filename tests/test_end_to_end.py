"""
End-to-End-Test der Live-Klassifikation (Aufgabe 5 / Issue #14).

Was hier getestet wird
----------------------
Der KOMPLETTE Inferenz-Pfad, so wie er im Live-Modus laeuft -- nur ohne Webcam,
gespeist aus einer echten Aufnahme:

    Aufnahme (detector-Frames)
        -> Preprocessor.step()      (sammelt Trajektorie, _to_features -> (N,3))
        -> HMMModule.step()         (laedt data/hmm.pkl, decision_function, argmax)
        -> vorhergesagter Buchstabe

Die bestehenden Tests decken nur EINZELNE Bausteine ab
(``test_live_segmentation`` stoppt beim Preprocessor, ``test_evaluate_classifier``
prueft nur Hilfsfunktionen). Dieser Test schliesst die Luecke: er beweist, dass
Detector-Output, Preprocessing und der trainierte Klassifikator im Zusammenspiel
tatsaechlich die richtige Geste erkennen -- der eigentliche Integrationsnachweis
des Systems.

Warum ueber HMMModule statt direkt ueber HMMClassifier?
------------------------------------------------------
Damit exakt der Live-Code-Pfad getestet wird: HMMModule.step() enthaelt die
Laengen-Normierung der Scores, die Argmax-Auswahl und die Schwellenwert-Logik.
Ein Test direkt auf dem Classifier wuerde diese Live-Schicht ueberspringen.

Bewusst ohne pytest-Fixtures -- laeuft auch direkt:

    python tests/test_end_to_end.py
"""

import glob
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
# Projekt-Root auf den Importpfad legen, damit das Paket ohne Installation laeuft.
sys.path.insert(0, str(ROOT))

from GestureRecognition.modules.preprocessor import Preprocessor  # noqa: E402
from GestureRecognition.modules.hiddenmarkov import HMMModule  # noqa: E402

ALPHABET = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
# Testperson mit vollem A-Z-Satz. Mehrere Aufnahmen je Buchstabe stabilisieren
# die Accuracy gegen den Zufall eines einzelnen (evtl. schlechten) Takes.
TEST_PERSON = "yannik"
SAMPLES_PER_LETTER = 2
# Gut trennbare Buchstaben (empirisch je 3/3 erkannt). Bewusst OHNE die
# form-aehnlichen Wackelkandidaten (V<->U, L<->F, N<->O): die koennen auf einem
# Einzelsample kippen, ohne dass der Live-Pfad kaputt ist.
STRONG_LETTERS = ["A", "B", "D", "E", "F", "I", "P", "T"]
STRONG_MIN_ACCURACY = 0.85
# Konservative Untergrenze: der komplette Trainingslauf misst ~0.93 Test-Accuracy,
# Einzelsamples liegen tiefer. 0.70 faengt echte Regressionen (z.B. Feature-Format-
# Drift zwischen Training und Live -> alles wird "?") sicher ab, ohne bei den
# bekannten schwachen Buchstaben (C/L/M/Q-Verwechslungen) zu flackern.
MIN_ACCURACY = 0.70


def _load_config() -> dict:
    with open(ROOT / "config.yml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _emit_gesture(detector_frames: list, cfg: dict) -> np.ndarray | None:
    """Schickt die detector-Frames einer Aufnahme durch den echten Preprocessor
    und gibt die laengste emittierte Trajektorie zurueck -- also die eigentliche
    Geste (kurze Fragmente werden dadurch ignoriert).

    Am Ende werden Frames ohne Hand angehaengt, damit der Preprocessor die laufende
    Geste abschliesst -- genau wie im Live-Betrieb, wenn die Hand weggenommen wird.
    """
    pre = Preprocessor()
    pre.start(cfg)
    hand_off = [{"detector": {"hands": []}} for _ in range(15)]
    emitted: list[np.ndarray] = []
    for frame in list(detector_frames) + hand_off:
        detector = frame.get("detector") if isinstance(frame, dict) else None
        step_data = dict(cfg)
        step_data["detector"] = detector
        out = pre.step(step_data)
        if out.get("preprocessor") is not None:
            emitted.append(np.asarray(out["preprocessor"], dtype=float))
    if not emitted:
        return None
    return max(emitted, key=len)


def _recordings_for(letter: str, person: str, limit: int) -> list[Path]:
    pattern = str(ROOT / f"recordings/{letter}/{letter}-{person}-*.pkl")
    return [Path(p) for p in sorted(glob.glob(pattern))[:limit]]


def _make_hmm_module(cfg: dict) -> HMMModule:
    """HMMModule instanziieren und starten (laedt data/hmm.pkl genau wie live)."""
    module = HMMModule()
    module.start(cfg)
    return module


def _classify(module: HMMModule, features: np.ndarray, cfg_raw: dict) -> dict:
    """Ein Frame durch HMMModule.step -- liefert das Klassifikations-Result."""
    result = module.step({"preprocessor": features, "config": cfg_raw})
    return result["markov"]


def test_modell_laedt_und_kennt_alle_buchstaben(ensure_hmm_model):
    """HMMModule.start() muss das Modell laden und mindestens alle 26 Buchstaben
    kennen. Zusaetzliche Klassen (z.B. eine selbst aufgenommene Geste) sind erlaubt
    -- das System soll ja um neue Gesten erweiterbar sein.
    """
    cfg_raw = _load_config()
    module = _make_hmm_module({"config": cfg_raw})
    assert module.classifier is not None, "HMMModule hat data/hmm.pkl nicht geladen."
    classes = {str(c) for c in module.classifier.classes_}
    fehlend = [b for b in ALPHABET if b not in classes]
    assert not fehlend, f"Buchstaben fehlen im Modell: {fehlend} (vorhanden: {sorted(classes)})"


def test_preprocessor_emittiert_trainingsformat(ensure_hmm_model):
    """Der Preprocessor muss live das exakte Trainingsformat (N, 3) liefern --
    (x, y, velocity). Ein anderes Format wuerde den Classifier live blind machen
    (Score -inf -> immer '?'). Regressionsschutz fuer den frueheren 4-Feature-Bug.
    """
    cfg_raw = _load_config()
    cfg = {"config": cfg_raw}
    files = _recordings_for("A", TEST_PERSON, limit=1)
    assert files, f"Testaufnahme 'A-{TEST_PERSON}-*' fehlt."
    recording = pickle.load(open(files[0], "rb"))
    features = _emit_gesture(recording.get("detector", []), cfg)
    assert features is not None, "Preprocessor hat gar keine Geste emittiert."
    assert features.ndim == 2 and features.shape[1] == 3, (
        f"Live-Feature-Format {features.shape} passt nicht zum Trainingsformat (N, 3)."
    )


def test_end_to_end_erkennt_gesten(ensure_hmm_model):
    """Kernnachweis: die komplette Live-Pipeline erkennt echte Aufnahmen ueber
    alle 26 Buchstaben hinweg mit einer Gesamt-Accuracy oberhalb der Schwelle.
    """
    cfg_raw = _load_config()
    cfg = {"config": cfg_raw}
    module = _make_hmm_module({"config": cfg_raw})

    correct = 0
    total = 0
    verwechslungen: list[str] = []
    fehlende: list[str] = []

    for letter in ALPHABET:
        files = _recordings_for(letter, TEST_PERSON, SAMPLES_PER_LETTER)
        if not files:
            fehlende.append(letter)
            continue
        for path in files:
            recording = pickle.load(open(path, "rb"))
            features = _emit_gesture(recording.get("detector", []), cfg)
            assert features is not None, f"'{path.name}': keine Geste emittiert."
            markov = _classify(module, features, cfg_raw)
            predicted = str(markov["best_label"])
            total += 1
            if predicted == letter:
                correct += 1
            else:
                verwechslungen.append(f"{letter}->{predicted}")

    assert not fehlende, f"Testaufnahmen fehlen fuer: {fehlende}"
    assert total >= 26, f"Zu wenige Testsequenzen ({total})."
    accuracy = correct / total
    assert accuracy >= MIN_ACCURACY, (
        f"Live-Erkennung zu schwach: {correct}/{total} = {accuracy:.3f} "
        f"< {MIN_ACCURACY:.2f}. Verwechslungen: {verwechslungen}"
    )


def test_starke_buchstaben_werden_zuverlaessig_erkannt(ensure_hmm_model):
    """Gut trennbare Buchstaben muessen ueber mehrere Samples zuverlaessig erkannt
    werden. Bewusst eine Trefferquote statt Einzelsample-Gleichheit: ein einzelnes
    Sample kann durch Formaehnlichkeit kippen (z.B. V<->U), ohne dass der Live-Pfad
    kaputt ist. Faellt die Quote unter die Schwelle, ist der Pfad grundlegend defekt.
    """
    cfg_raw = _load_config()
    cfg = {"config": cfg_raw}
    module = _make_hmm_module({"config": cfg_raw})

    correct = 0
    total = 0
    fehlklassifikationen: list[str] = []
    for letter in STRONG_LETTERS:
        for path in _recordings_for(letter, TEST_PERSON, SAMPLES_PER_LETTER):
            recording = pickle.load(open(path, "rb"))
            features = _emit_gesture(recording.get("detector", []), cfg)
            assert features is not None, f"'{path.name}': keine Geste emittiert."
            predicted = str(_classify(module, features, cfg_raw)["best_label"])
            total += 1
            if predicted == letter:
                correct += 1
            else:
                fehlklassifikationen.append(f"{letter}->{predicted}")

    assert total > 0, "Keine Testaufnahmen fuer die starken Buchstaben gefunden."
    accuracy = correct / total
    assert accuracy >= STRONG_MIN_ACCURACY, (
        f"Starke Buchstaben nur {correct}/{total} = {accuracy:.2f} < "
        f"{STRONG_MIN_ACCURACY:.2f}. Fehlklassifikationen: {fehlklassifikationen}"
    )


if __name__ == "__main__":
    # Standalone: Modell sicherstellen (trainiert bei Bedarf), dann Tests direkt
    # aufrufen. Die ensure_hmm_model-Fixture gibt es nur unter pytest -- hier
    # bereiten wir das Modell mit derselben Logik selbst vor.
    from conftest import prepare_hmm_model

    marker = prepare_hmm_model()
    test_modell_laedt_und_kennt_alle_buchstaben(marker)
    test_preprocessor_emittiert_trainingsformat(marker)
    test_end_to_end_erkennt_gesten(marker)
    test_starke_buchstaben_werden_zuverlaessig_erkannt(marker)
    print("OK: End-to-End-Live-Klassifikation erkennt Gesten ueber die volle Pipeline.")
