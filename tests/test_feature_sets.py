"""
Tests fuer die Feature-Formate aus Issue #59.

Sichert ab:
  1. Beide Feature-Sets liefern die richtige Spaltenzahl.
  2. In der Alternative passt speed exakt zu (dx, dy) -- speed = |(dx, dy)|.
  3. Live-Pfad und Training koennen nicht auseinanderlaufen: der Preprocessor
     benutzt _to_features ohne eigenes feature_set, folgt also automatisch der
     zentralen Wahl FEATURE_SET.

Bewusst ohne pytest-Setup -- direkt lauffaehig:

    python tests/test_feature_sets.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from GestureRecognition.labeling import (  # noqa: E402
    FEATURE_SET,
    RESAMPLE_LENGTH,
    _to_features,
)


def _beispiel_trajektorie(n=60) -> np.ndarray:
    """Ein Halbkreis als realistische Beispiel-Geste."""
    t = np.linspace(0, np.pi, n)
    return np.column_stack([0.5 + 0.2 * np.cos(t), 0.5 + 0.2 * np.sin(t)])


def test_spaltenzahl():
    traj = _beispiel_trajektorie()
    assert _to_features(traj, feature_set="xyv").shape == (RESAMPLE_LENGTH, 3)
    assert _to_features(traj, feature_set="xydxdyv").shape == (RESAMPLE_LENGTH, 5)


def test_visualisierung_kann_originale_laenge_behalten():
    traj = _beispiel_trajektorie(n=73)
    assert _to_features(traj, resample_length=None).shape == (73, 3)


def test_speed_passt_zu_richtung():
    feats = _to_features(_beispiel_trajektorie(), feature_set="xydxdyv")
    dx, dy, speed = feats[:, 2], feats[:, 3], feats[:, 4]
    assert np.allclose(np.hypot(dx, dy), speed), "speed muss |(dx, dy)| sein"
    # Erster Frame hat keinen Vorgaenger -> ueberall 0.
    assert dx[0] == dy[0] == speed[0] == 0.0


def test_default_folgt_zentraler_wahl():
    traj = _beispiel_trajektorie()
    default = _to_features(traj)
    explizit = _to_features(traj, feature_set=FEATURE_SET)
    assert default.shape == explizit.shape
    assert np.allclose(default, explizit), (
        "_to_features() ohne Argument muss der zentralen Wahl FEATURE_SET folgen "
        "-- sonst koennen Training und Live auseinanderlaufen."
    )


def test_unbekanntes_set_wird_abgelehnt():
    try:
        _to_features(_beispiel_trajektorie(), feature_set="quatsch")
    except ValueError:
        return
    raise AssertionError("unbekanntes feature_set muss ValueError ausloesen")


if __name__ == "__main__":
    test_spaltenzahl()
    test_visualisierung_kann_originale_laenge_behalten()
    test_speed_passt_zu_richtung()
    test_default_folgt_zentraler_wahl()
    test_unbekanntes_set_wird_abgelehnt()
    print("OK: Feature-Sets konsistent (Issue #59)")
