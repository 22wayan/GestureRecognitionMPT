"""
Selbstcheck fuer die Buchstaben-Auswahl der freien Aufnahme.

Bewusst ohne pytest-Setup — direkt lauffaehig:

    python tests/test_collect_letters.py

Getestet wird nur die reine Logik (_parse_letters, _format_letters). Die
Kamera-/Aufnahme-Integration wird per End-to-End-Lauf verifiziert.
"""

import sys
from pathlib import Path

# Projekt-Root auf den Importpfad legen, damit das Paket ohne Installation laeuft.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from GestureRecognition.labeling import _parse_letters, _format_letters  # noqa: E402


def test_parse_letters_mit_kommas():
    assert _parse_letters("Q,C,W,O,M") == ["Q", "C", "W", "O", "M"]


def test_parse_letters_ohne_trenner():
    assert _parse_letters("qcwom") == ["Q", "C", "W", "O", "M"]


def test_parse_letters_leerzeichen_und_kleinschreibung():
    assert _parse_letters("q c w") == ["Q", "C", "W"]


def test_parse_letters_entfernt_duplikate_behaelt_reihenfolge():
    assert _parse_letters("Q, C, Q, W, C") == ["Q", "C", "W"]


def test_parse_letters_ignoriert_ungueltige_zeichen():
    assert _parse_letters("Q1! C-3") == ["Q", "C"]


def test_parse_letters_leer_ergibt_leere_liste():
    assert _parse_letters("") == []
    assert _parse_letters("123 ...") == []


def test_format_letters_zusammenhaengender_bereich():
    assert _format_letters(["A", "B", "C", "D"]) == "A-D"
    assert _format_letters([chr(c) for c in range(ord("A"), ord("Z") + 1)]) == "A-Z"


def test_format_letters_freie_auswahl():
    assert _format_letters(["Q", "C", "W", "O", "M"]) == "Q, C, W, O, M"
    # Nicht zusammenhaengend -> Liste, kein Bereich
    assert _format_letters(["A", "C", "E"]) == "A, C, E"


if __name__ == "__main__":
    tests = sorted(
        (name, fn)
        for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    )
    for name, fn in tests:
        fn()
        print(f"  ok: {name}")
    print(f"Alle {len(tests)} Tests bestanden.")
