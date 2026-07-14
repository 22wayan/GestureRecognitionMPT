"""
End-to-End-Test mit einer WIRKLICH UNBEKANNTEN Person (Issue #61).

Abgrenzung zu test_end_to_end.py
--------------------------------
``test_end_to_end.py`` testet die Pipeline mit einer BEKANNTEN Person
(yannik): deren Aufnahmen stecken auch im Training von ``data/hmm.pkl``.
Das beweist, dass die Pipeline funktioniert -- aber nicht, dass das System
auf eine fremde Person generalisiert (das Pruefer-Szenario).

Dieser Test schliesst genau diese Luecke:

  1. Das Modell wird OHNE die Testperson trainiert
     (``conftest.ensure_holdout_model``; Aufnahmen ohne Personen-Tag fliegen
     ebenfalls raus, weil bei ihnen nicht beweisbar ist, wer sie gemacht hat).
  2. Die gespeicherten Detector-Frames der Testperson laufen durch den ECHTEN
     ``Preprocessor`` und das ECHTE ``HMMModule`` -- exakt der Live-Pfad.
  3. Bewertet wird das tatsaechlich ANGEZEIGTE ``label`` (inklusive der
     Schwellenwert-Logik, die "?" ausgeben kann) -- nicht der interne Tipp
     ``best_label``. "?" wird als eigenes Ergebnis gezaehlt und berichtet.

Bewusst ohne pytest-Zwang -- laeuft auch direkt:

    python tests/test_end_to_end_unknown_person.py
"""

import logging
import pickle
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Projekt-Root auf den Importpfad legen, damit das Paket ohne Installation laeuft.
sys.path.insert(0, str(ROOT))

from conftest import HOLDOUT_MANIFEST_PATH, HOLDOUT_MODEL_PATH, HOLDOUT_PERSON  # noqa: E402
from test_end_to_end import ALPHABET, _emit_gesture, _load_config, _recordings_for  # noqa: E402

from GestureRecognition.modules.hiddenmarkov import HMMModule  # noqa: E402
from GestureRecognition.visualization import _person_from_filename  # noqa: E402

_log = logging.getLogger("tests.unknown_person")

# Mehrere Takes pro Buchstabe glaetten den Zufall einzelner (schlechter)
# Aufnahmen: 3 x 26 = 78 Testgesten.
SAMPLES_PER_LETTER = 3

# Schwellen -- begruendet durch eine GEMESSENE Baseline, nicht geraten.
# Messlauf 2026-07-14 (Modell ohne wayan: 1060 Sequenzen / 27 Klassen,
# n_components=10, random_state=42; 78 Testgesten von wayan):
#   62/78 = 79.5% korrekt angezeigt, 9/78 = 11.5% als "?", 7 Verwechslungen
#   (Q->O 2x, F->P, G->C, J->O, P->O, Q->D).
# MIN_ACCURACY = 0.65: Baseline minus ~15 Prozentpunkte. Ein einzelner
# gekippter Take bewegt die Quote um 1/78 = 1.3pp, die Schwelle liegt also weit
# ausserhalb normaler Schwankung -- aber weit OBERHALB der bekannten
# Fehlerbilder (Feature-Format-Drift oder invertierte Hysterese druecken die
# Quote Richtung 0%, weil fast alles "?" wird).
MIN_ACCURACY = 0.65
# MAX_UNKNOWN_RATE = 0.30: knapp das Dreifache der gemessenen 11.5%. Faengt den
# klassischen Totalausfall ("alles wird ?") sofort, ohne bei ein paar
# zusaetzlichen unsicheren Takes zu flackern.
MAX_UNKNOWN_RATE = 0.30


def _make_holdout_module(cfg_raw: dict) -> HMMModule:
    """HMMModule mit dem Hold-out-Modell (ohne Testperson) starten."""
    module = HMMModule(model_path=str(HOLDOUT_MODEL_PATH))
    module.start({"config": cfg_raw})
    assert module.classifier is not None, (
        f"Hold-out-Modell nicht geladen: {HOLDOUT_MODEL_PATH}"
    )
    return module


def _evaluate_unknown_person() -> dict:
    """Laesst alle Testgesten der unbekannten Person durch den Live-Pfad laufen.

    Rueckgabe: {"total", "correct", "unknown", "confusions" (Counter),
    "accuracy", "unknown_rate", "fehlende_buchstaben"}.
    """
    cfg_raw = _load_config()
    cfg = {"config": cfg_raw}
    module = _make_holdout_module(cfg_raw)

    total = 0
    correct = 0
    unknown = 0
    confusions: Counter[str] = Counter()
    fehlende: list[str] = []

    for letter in ALPHABET:
        files = _recordings_for(letter, HOLDOUT_PERSON, SAMPLES_PER_LETTER)
        if not files:
            fehlende.append(letter)
            continue
        for path in files:
            recording = pickle.load(open(path, "rb"))
            features = _emit_gesture(recording.get("detector", []), cfg)
            assert features is not None, f"'{path.name}': keine Geste emittiert."
            markov = module.step({"preprocessor": features, "config": cfg_raw})["markov"]
            # Das ANGEZEIGTE Label bewerten (kann "?" sein) -- nicht best_label.
            angezeigt = str(markov["label"])
            total += 1
            if angezeigt == letter:
                correct += 1
            elif markov["unknown"]:
                unknown += 1
                confusions[f"{letter}->?"] += 1
            else:
                confusions[f"{letter}->{angezeigt}"] += 1

    return {
        "total": total,
        "correct": correct,
        "unknown": unknown,
        "confusions": confusions,
        "accuracy": correct / total if total else 0.0,
        "unknown_rate": unknown / total if total else 0.0,
        "fehlende_buchstaben": fehlende,
    }


def _bericht(report: dict) -> str:
    """Formatiert das Ergebnis als lesbaren Bericht (fuer Log und Assert-Meldung)."""
    verwechslungen = ", ".join(
        f"{paar} ({anzahl}x)" for paar, anzahl in report["confusions"].most_common()
    ) or "keine"
    return (
        f"Unbekannte Person '{HOLDOUT_PERSON}': "
        f"{report['correct']}/{report['total']} korrekt angezeigt "
        f"= {report['accuracy']:.1%} | "
        f"als '?' angezeigt: {report['unknown']} ({report['unknown_rate']:.1%}) | "
        f"Verwechslungen: {verwechslungen}"
    )


def test_testperson_nachweislich_nicht_im_training(ensure_holdout_model):
    """Akzeptanzkriterium: die Testperson kommt in KEINER Trainingsaufnahme vor.

    Das Manifest haelt fest, mit welchen Dateien das Hold-out-Modell trainiert
    wurde. Wir pruefen beides: (a) keine Trainingsdatei traegt den Tag der
    Testperson (auch keine ohne Tag -- die sind komplett ausgeschlossen), und
    (b) es gibt ueberhaupt Aufnahmen der Testperson (sonst waere der Hold-out
    ein leeres Versprechen).
    """
    import json

    manifest = json.loads(HOLDOUT_MANIFEST_PATH.read_text(encoding="utf-8"))
    train_files = manifest["train_files"]
    assert manifest["holdout_person"] == HOLDOUT_PERSON
    assert len(train_files) > 0, "Manifest enthaelt keine Trainingsdateien."

    personen = [_person_from_filename(ROOT / f) for f in train_files]
    assert all(p is not None for p in personen), (
        "Trainingsdatei ohne Personen-Tag im Hold-out-Training -- nicht beweisbar."
    )
    assert HOLDOUT_PERSON not in personen, (
        f"Aufnahmen von '{HOLDOUT_PERSON}' im Hold-out-Training gefunden!"
    )

    testdateien = list((ROOT / "recordings").glob(f"*/*-{HOLDOUT_PERSON}-*.pkl"))
    assert len(testdateien) >= 26, (
        f"Zu wenige Aufnahmen der Testperson '{HOLDOUT_PERSON}' ({len(testdateien)})."
    )


def test_end_to_end_unbekannte_person(ensure_holdout_model):
    """Kernnachweis der Generalisierung: die komplette Live-Pipeline erkennt die
    Gesten einer Person, die das Modell nie gesehen hat -- bewertet am
    tatsaechlich angezeigten Label (inklusive '?').
    """
    report = _evaluate_unknown_person()
    _log.warning(_bericht(report))

    assert not report["fehlende_buchstaben"], (
        f"Testaufnahmen von '{HOLDOUT_PERSON}' fehlen fuer: {report['fehlende_buchstaben']}"
    )
    assert report["total"] >= 26, f"Zu wenige Testsequenzen ({report['total']})."
    assert report["accuracy"] >= MIN_ACCURACY, (
        f"Generalisierung zu schwach: {_bericht(report)} "
        f"(Schwelle: {MIN_ACCURACY:.2f})"
    )
    assert report["unknown_rate"] <= MAX_UNKNOWN_RATE, (
        f"Zu oft '?' angezeigt: {_bericht(report)} "
        f"(Schwelle: {MAX_UNKNOWN_RATE:.2f})"
    )


if __name__ == "__main__":
    # Standalone: beide Modelle sicherstellen, dann Tests direkt aufrufen und
    # den Bericht ausgeben (gleiche Logik wie unter pytest).
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    from conftest import prepare_holdout_model

    marker = prepare_holdout_model()
    test_testperson_nachweislich_nicht_im_training(marker)
    report = _evaluate_unknown_person()
    print(_bericht(report))
    test_end_to_end_unbekannte_person(marker)
    print("OK: End-to-End-Pfad generalisiert auf eine unbekannte Person.")
