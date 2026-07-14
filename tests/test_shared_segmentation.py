"""
Regressionstest fuer die GEMEINSAME Gesten-Segmentierung (Issue #58).

Hintergrund: Frueher segmentierten Training und Live-Modus unterschiedlich --
das Training nahm ueber _extract_trajectory alle Frames mit Hand (inklusive
"Hand zum Startpunkt fuehren"), der Live-Preprocessor schnitt per
Geschwindigkeits-Hysterese nur die eigentliche Bewegung aus. Das Modell lernte
dadurch andere Sequenzen, als es live bewerten musste.

Seit Issue #58 lebt die Segmentierung genau einmal, in
GestureRecognition.labeling.GestureSegmenter. Dieser Test stellt sicher, dass
Live-Pfad (Preprocessor) und Offline-Pfad (segment_trajectory, wie im
Datensatzbau) aus DENSELBEN Detector-Frames DIESELBEN Sequenzen erzeugen --
und schlaegt fehl, sobald einer der beiden Pfade eigene Logik bekommt.

Bewusst ohne pytest-Setup -- direkt lauffaehig:

    python tests/test_shared_segmentation.py
"""

import glob
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from GestureRecognition.labeling import (  # noqa: E402
    _extract_trajectory,
    _to_features,
    segment_trajectory,
)
from GestureRecognition.modules.preprocessor import Preprocessor  # noqa: E402


def _load_config() -> dict:
    with open(ROOT / "config.yml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _segmentation_params(cfg: dict) -> dict:
    """Preprocessor-Konfiguration in die Parameter von segment_trajectory uebersetzen."""
    pre = cfg["preprocessor"]
    return {
        "min_speed": pre["min_speed_corner"],
        "reset_speed": pre["reset_speed_corner"],
        "stop_hold": pre["stop_hold"],
        "max_lost": pre["max_lost"],
        "min_steps": pre["min_steps"],
        "buffer_size": pre["buffer_size"],
    }


def _run_live_preprocessor(frames: list, cfg: dict) -> list:
    """Schickt die detector-Frames durch den echten Live-Preprocessor und gibt
    alle emittierten Feature-Trajektorien zurueck."""
    pre = Preprocessor()
    pre.start({"config": cfg})
    emitted = []
    for frame in frames:
        det = frame.get("detector") if isinstance(frame, dict) else None
        out = pre.step({"config": cfg, "detector": det})
        if out.get("preprocessor") is not None:
            emitted.append(np.asarray(out["preprocessor"]))
    return emitted


def _sample_recordings() -> list:
    """Ein paar echte Aufnahmen verschiedener Buchstaben als Testmaterial."""
    paths = []
    for letter in ["O", "L", "Z", "M"]:
        matches = sorted(glob.glob(str(ROOT / f"recordings/{letter}/{letter}-yannik-*.pkl")))
        assert matches, f"Testaufnahme fuer '{letter}' fehlt"
        paths.append(matches[0])
    return paths


def test_live_und_offline_segmentieren_identisch():
    """Dieselben Detector-Frames muessen im Live-Pfad und im Datensatzbau zu
    denselben Gestensequenzen fuehren (Akzeptanzkriterium aus Issue #58)."""
    cfg = _load_config()
    params = _segmentation_params(cfg)
    # Hand-Wegnehmen am Ende simulieren, wie im Live-Betrieb ueblich.
    hand_off = [{"detector": {"hands": []}} for _ in range(params["max_lost"] + 2)]

    for path in _sample_recordings():
        with open(path, "rb") as f:
            recording = pickle.load(f)
        frames = list(recording.get("detector", [])) + hand_off

        # Live-Pfad: der echte Preprocessor, Frame fuer Frame.
        live = _run_live_preprocessor(frames, cfg)

        # Offline-Pfad: genau wie clean_recordings -- Trajektorie extrahieren,
        # dann die gemeinsame Segmentierung anwenden, dann Features.
        raw = _extract_trajectory(recording, cfg["preprocessor"]["finger_idx"])
        assert raw is not None, f"{path}: keine Trajektorie extrahierbar"
        offline = [_to_features(seg) for seg in segment_trajectory(raw, **params)]

        name = Path(path).name
        assert len(live) == len(offline), (
            f"{name}: Live emittiert {len(live)} Segment(e), "
            f"Offline {len(offline)} -- die Pfade sind auseinandergelaufen!"
        )
        for i, (a, b) in enumerate(zip(live, offline)):
            assert a.shape == b.shape, (
                f"{name}, Segment {i}: Form live {a.shape} vs offline {b.shape}"
            )
            assert np.allclose(a, b), (
                f"{name}, Segment {i}: Werte von Live- und Offline-Segmentierung "
                "unterscheiden sich."
            )


def test_anfahrt_landet_nicht_im_segment():
    """Vorbereitungsbewegung vor der Geste darf nicht im Segment landen: eine
    langsame Anfahrt gefolgt von schneller Bewegung ergibt ein Segment, das
    erst bei der schnellen Bewegung beginnt."""
    cfg = _load_config()
    params = _segmentation_params(cfg)

    slow = params["reset_speed"] * 0.5   # deutlich unter der Stopp-Schwelle
    fast = params["min_speed"] * 3.0     # deutlich ueber der Start-Schwelle

    points = []
    x = 0.1
    # 30 Frames langsame "Anfahrt" (unter jeder Schwelle).
    for _ in range(30):
        x += slow
        points.append([x, 0.5])
    start_of_gesture = len(points)
    # 40 Frames echte Geste.
    for _ in range(40):
        x += fast
        points.append([x, 0.5])
    # Ende: Hand weg.
    traj = np.array(points + [[np.nan, np.nan]] * (params["max_lost"] + 2))

    segments = segment_trajectory(traj, **params)
    assert len(segments) == 1, f"erwartet 1 Segment, bekommen {len(segments)}"
    segment = segments[0]
    # Das Segment beginnt am Uebergang zur schnellen Bewegung (die letzten zwei
    # Punkte davor gehoeren als Startpunkt dazu), NICHT am Anfang der Anfahrt.
    expected_max = 40 + 2
    assert len(segment) <= expected_max, (
        f"Anfahrt im Segment gelandet: {len(segment)} Punkte "
        f"(erwartet <= {expected_max}) -- Segment beginnt zu frueh."
    )
    # Der erste Segment-Punkt muss (bis auf die zwei Startpunkte) hinter der
    # Anfahrt liegen.
    first_x = segment[0, 0]
    anfahrt_end_x = points[start_of_gesture - 1][0]
    assert first_x >= anfahrt_end_x - 2 * slow - 1e-9, (
        "Segment enthaelt Anfahrts-Punkte."
    )


if __name__ == "__main__":
    test_live_und_offline_segmentieren_identisch()
    test_anfahrt_landet_nicht_im_segment()
    print("OK: Training und Live segmentieren identisch (Issue #58)")
