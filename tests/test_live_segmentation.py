"""
Regressionstest fuer die Live-Gesten-Segmentierung (Issue #15).

Hintergrund: Frueher war die Hysterese in der config.yml invertiert
(Start-Schwelle < Stopp-Schwelle). Dadurch wurde jede Geste bei der kleinsten
Tempo-Delle sofort beendet und in winzige 15-Punkt-Fetzen zerhackt. Der
Klassifikator sah nur Teil-Gesten und gab fast immer "?" aus -- der Live-Modus
zeigte praktisch nie einen Buchstaben.

Dieser Test verhindert, dass das zurueckkommt. Er braucht KEIN trainiertes
Modell (nur den Preprocessor und eine mitgelieferte Aufnahme).

Bewusst ohne pytest-Setup -- direkt lauffaehig:

    python tests/test_live_segmentation.py
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


def _load_config() -> dict:
    with open(ROOT / "config.yml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run_preprocessor(frames: list, cfg: dict) -> list:
    """Schickt die detector-Frames einer Aufnahme durch den echten Preprocessor
    und gibt die Laengen aller emittierten Trajektorien zurueck."""
    pre = Preprocessor()
    pre.start(cfg)
    lengths = []
    for frame in frames:
        det = frame.get("detector") if isinstance(frame, dict) else None
        data = dict(cfg)
        data["detector"] = det
        out = pre.step(data)
        if out.get("preprocessor") is not None:
            lengths.append(len(np.asarray(out["preprocessor"])))
    return lengths


def test_hysterese_nicht_invertiert():
    """Start-Schwelle MUSS ueber der Stopp-Schwelle liegen (echte Hysterese)."""
    pre = _load_config()["preprocessor"]
    assert pre["min_speed_corner"] > pre["reset_speed_corner"], (
        "Hysterese invertiert: min_speed_corner (Start) muss > reset_speed_corner "
        "(Stopp) sein, sonst zerhackt die Segmentierung jede Geste in Fetzen."
    )


def test_ganze_geste_wird_erfasst():
    """Eine Geste muss als ZUSAMMENHAENGENDE Trajektorie erfasst werden, nicht
    in viele 15-Punkt-Fetzen zerfallen."""
    cfg = {"config": _load_config()}
    # Ein paar sauber aufgenommene Buchstaben; am Ende simulieren wir das
    # Hand-Wegnehmen (so wird auch im Live-Betrieb die Geste beendet).
    hand_off = [{"detector": {"hands": []}} for _ in range(15)]
    for letter in ["O", "L", "Z"]:
        matches = sorted(glob.glob(str(ROOT / f"recordings/{letter}/{letter}-yannik-*.pkl")))
        assert matches, f"Testaufnahme fuer '{letter}' fehlt"
        with open(matches[0], "rb") as f:
            recording = pickle.load(f)
        frames = list(recording.get("detector", [])) + hand_off
        hand_frames = sum(
            1
            for fr in frames
            if isinstance(fr, dict)
            and isinstance(fr.get("detector"), dict)
            and fr["detector"].get("hands")
        )

        lengths = _run_preprocessor(frames, cfg)
        assert lengths, f"'{letter}': Preprocessor hat gar keine Geste emittiert"

        longest = max(lengths)
        # Die laengste erfasste Geste soll einen Grossteil der Hand-Frames
        # abdecken -- ein 15-Punkt-Fetzen (altes, kaputtes Verhalten) wuerde hier
        # klar durchfallen.
        assert longest >= 0.6 * hand_frames, (
            f"'{letter}': Geste zu fragmentiert -- laengster Emit {longest} von "
            f"{hand_frames} Hand-Frames (erwartet >= {0.6 * hand_frames:.0f})."
        )


if __name__ == "__main__":
    test_hysterese_nicht_invertiert()
    test_ganze_geste_wird_erfasst()
    print("OK: Live-Segmentierung erfasst ganze Gesten (Issue #15)")
