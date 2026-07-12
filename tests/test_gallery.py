"""
Selbstcheck fuer die Aufnahme-Galerie (replay_recordings).

Bewusst ohne pytest-Setup — direkt lauffaehig:

    python tests/test_gallery.py

Getestet wird die knifflige Logik: bekommt jede Aufnahme den richtigen Status
(ok / kurz / Sprung / keine Hand), und legt replay_recordings die erwarteten
PNG-Dateien + eine korrekte Zusammenfassung an. Das eigentliche Aussehen der
Plots wird per End-to-End-Lauf (visualize.py) geprueft.
"""

import pickle
import sys
import tempfile
from pathlib import Path

# Headless: kein Fenster oeffnen, nur in PNG rendern. Muss VOR dem Import von
# visualization passieren (dort wird matplotlib.pyplot geladen).
import matplotlib

matplotlib.use("Agg")

# Projekt-Root auf den Importpfad legen, damit das Paket ohne Installation laeuft.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from GestureRecognition.visualization import (  # noqa: E402
    _STATUS_KEINE_HAND,
    _STATUS_KURZ,
    _STATUS_OK,
    _STATUS_SPRUNG,
    _load_recordings_with_status,
    replay_recordings,
)

FINGER = 8  # Zeigefingerspitze (Standard-Landmark)


def _frame(point):
    """Baut einen einzelnen Aufnahme-Frame im neuen Dict-Format.

    ``point`` ist ein (x, y)-Tupel oder ``None`` (keine Hand in diesem Frame).
    """
    if point is None:
        return {"detector": {"hands": []}}
    # Wir brauchen mindestens FINGER+1 Landmarken; nur Index FINGER wird gelesen.
    landmarks = [{"x": point[0], "y": point[1]} for _ in range(FINGER + 1)]
    return {"detector": {"hands": [{"landmarks": landmarks}]}}


def _write_recording(path, points):
    """Schreibt eine .pkl-Aufnahme aus einer Liste von (x, y)-Punkten/None."""
    recording = {"detector": [_frame(p) for p in points]}
    with open(path, "wb") as f:
        pickle.dump(recording, f)


def _build_dataset(root):
    """Legt einen Mini-Datensatz mit genau einem Beispiel je Status an."""
    a = Path(root) / "A"
    a.mkdir(parents=True)

    # ok: 20 Frames, ruhige Bewegung (kein Sprung, lang genug)
    _write_recording(a / "A-ok.pkl", [(0.5 + 0.005 * i, 0.5 + 0.005 * i) for i in range(20)])
    # kurz: nur 5 Frames (unter min_length=15)
    _write_recording(a / "A-kurz.pkl", [(0.5, 0.5)] * 5)
    # Sprung: mittendrin ein grosser Satz (> max_jump=0.15)
    _write_recording(a / "A-sprung.pkl", [(0.5, 0.5)] * 10 + [(0.95, 0.05)] + [(0.5, 0.5)] * 9)
    # keine Hand: 20 Frames ganz ohne erkannte Hand
    _write_recording(a / "A-leer.pkl", [None] * 20)
    return Path(root)


def test_load_status_flags():
    with tempfile.TemporaryDirectory() as tmp:
        root = _build_dataset(tmp)
        galerie = _load_recordings_with_status(root)

        status = {e["name"]: e["status"] for e in galerie["A"]}
        assert status["A-ok"] == _STATUS_OK
        assert status["A-kurz"] == _STATUS_KURZ
        assert status["A-sprung"] == _STATUS_SPRUNG
        assert status["A-leer"] == _STATUS_KEINE_HAND


def test_load_keine_hand_hat_keine_trajektorie():
    with tempfile.TemporaryDirectory() as tmp:
        root = _build_dataset(tmp)
        galerie = _load_recordings_with_status(root)

        leer = next(e for e in galerie["A"] if e["name"] == "A-leer")
        assert leer["traj"] is None
        # Eine gute Aufnahme hat dagegen eine anzeigbare Bahn.
        ok = next(e for e in galerie["A"] if e["name"] == "A-ok")
        assert ok["traj"] is not None and len(ok["traj"]) > 0


def test_replay_erzeugt_png_und_zusammenfassung():
    with tempfile.TemporaryDirectory() as tmp:
        root = _build_dataset(tmp)
        out = Path(tmp) / "galerie"
        zusammenfassung = replay_recordings(root, output_dir=out)

        # Pro Buchstabe eine PNG-Datei
        assert (out / "gallery_A.png").exists()
        # Zaehlungen stimmen: 4 gesamt, 1 ok, 3 markiert
        assert zusammenfassung["A"]["gesamt"] == 4
        assert zusammenfassung["A"]["ok"] == 1
        assert zusammenfassung["A"]["markiert"] == 3


if __name__ == "__main__":
    test_load_status_flags()
    test_load_keine_hand_hat_keine_trajektorie()
    test_replay_erzeugt_png_und_zusammenfassung()
    print("OK: alle Selbstchecks bestanden")
