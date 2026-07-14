"""
Erzeugt Demo-GIFs der Klassifikation (fuer Doku & Praesentation, Issue #16).

Was das Skript macht (einfach erklaert)
---------------------------------------
Wir nehmen echte Aufnahmen aus ``recordings/`` und spielen sie Frame fuer
Frame ab -- so wie im Replay-Modus. Dabei zeichnen wir:

  - die 21 Hand-Punkte (gruen),
  - die Spur der Zeigefingerspitze (orange),
  - und am Ende den vom Modell erkannten Buchstaben (gross im Bild).

Der erkannte Buchstabe kommt aus GENAU derselben Pipeline wie im Live-Modus:
``_to_features`` (gleiche Punktanzahl + normalisieren + Geschwindigkeit) und
dann der trainierte ``HMMClassifier`` aus ``data/hmm.pkl``.

Benutzung
---------
    python train.py                     # einmal vorher: Modell bauen
    python demo_gif.py                  # Standard: Buchstaben A, M, W
    python demo_gif.py --letters B,X,Z  # eigene Auswahl
Die GIFs landen unter ``images/demo_<Buchstabe>.gif``.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from GestureRecognition.hmmclassifier import HMMClassifier
from GestureRecognition.labeling import _to_features

# Groesse der GIF-Bilder (klein halten, damit die Dateien nicht riesig werden).
BREITE, HOEHE = 480, 270
FINGER_IDX = 8          # Zeigefingerspitze (MediaPipe-Punkt Nummer 8)
# Schwellen wie im Live-Modus (config.yml): darunter sagt das Modell lieber "?".
SCORE_SCHWELLE = -20.0
MARGIN_SCHWELLE = 0.1


def lade_landmarks(pfad):
    """Liest aus einer Aufnahme fuer jeden Frame die 21 Hand-Punkte.

    Gibt eine Liste zurueck: pro Frame entweder eine Liste von (x, y)-Punkten
    oder None (wenn in dem Frame keine Hand erkannt wurde). Es gibt zwei
    Aufnahme-Formate (alt und neu) -- wir unterstuetzen beide.
    """
    with open(pfad, "rb") as f:
        aufnahme = pickle.load(f)

    frames = []
    for frame in aufnahme.get("detector", []):
        if not isinstance(frame, dict):
            frames.append(None)
            continue
        det = frame.get("detector")
        punkte = None
        if hasattr(det, "hand_landmarks"):
            # Altes Format: MediaPipe-Objekt
            if len(det.hand_landmarks) > 0:
                punkte = [(lm.x, lm.y) for lm in det.hand_landmarks[0]]
        elif isinstance(det, dict):
            # Neues Format: einfaches Dict
            haende = det.get("hands", [])
            if haende:
                punkte = [(lm["x"], lm["y"]) for lm in haende[0]["landmarks"]]
        frames.append(punkte)
    return frames


def klassifiziere(spur, modell):
    """Fragt das Modell: welcher Buchstabe ist diese Spur?

    Macht dieselben Schritte wie der Live-Modus: erst _to_features, dann
    Score pro Buchstabe. Ist der Score zu schlecht oder der Abstand zum
    zweitbesten zu klein, geben wir "?" zurueck (wie live).
    """
    features = _to_features(np.array(spur, dtype=float))
    scores = modell.decision_function(features, [len(features)])[0]
    # Score durch die Laenge teilen, damit die Zahl vergleichbar ist (wie live).
    scores = scores / len(features)
    beste_idx = int(np.argmax(scores))
    bester = float(scores[beste_idx])
    sortiert = np.sort(scores)[::-1]
    abstand = float(sortiert[0] - sortiert[1]) if len(sortiert) > 1 else 999.0
    label = modell.classes_[beste_idx]
    if bester < SCORE_SCHWELLE or abstand < MARGIN_SCHWELLE:
        label = "?"
    return label, bester


def male_frame(punkte, spur, text, text_gross=None):
    """Malt ein einzelnes GIF-Bild: Hand-Punkte, Spur und Text."""
    bild = Image.new("RGB", (BREITE, HOEHE), (25, 25, 30))
    d = ImageDraw.Draw(bild)

    # Spur der Fingerspitze (orange Linie)
    if len(spur) > 1:
        pixel = [(x * BREITE, y * HOEHE) for x, y in spur]
        d.line(pixel, fill=(255, 140, 0), width=4, joint="curve")

    # Die 21 Hand-Punkte (gruen)
    if punkte:
        for x, y in punkte:
            d.ellipse([x * BREITE - 3, y * HOEHE - 3, x * BREITE + 3, y * HOEHE + 3],
                      fill=(0, 220, 0))

    # Text oben links (kleine Statuszeile)
    try:
        font = ImageFont.load_default(size=16)
        font_gross = ImageFont.load_default(size=44)
    except TypeError:
        # Aeltere Pillow-Version: Standard-Schrift ohne Groesse
        font = ImageFont.load_default()
        font_gross = font
    d.text((10, 8), text, fill=(255, 255, 255), font=font)

    # Grosser Text unten links (das Ergebnis am Ende) -- bewusst nicht in der
    # Mitte, damit er die gemalte Spur nicht verdeckt.
    if text_gross:
        d.text((16, HOEHE - 60), text_gross,
               fill=(80, 255, 120), font=font_gross)
    return bild


def gif_fuer_aufnahme(pfad, modell, ziel):
    """Baut aus EINER Aufnahme ein GIF mit Spur + Ergebnis.

    Gibt das erkannte Label zurueck (oder None, wenn die Aufnahme unbrauchbar
    ist). So kann der Aufrufer ein Beispiel aussuchen, das korrekt erkannt wird.
    """
    frames_landmarks = lade_landmarks(pfad)

    # Leerlauf abschneiden: Frames ohne Hand am Anfang und Ende interessieren
    # im GIF nicht (sonst sieht man sekundenlang nur schwarzen Hintergrund).
    mit_hand = [i for i, p in enumerate(frames_landmarks) if p]
    if len(mit_hand) < 2:
        print(f"  {pfad.name}: zu wenig Handpunkte, uebersprungen")
        return None
    frames_landmarks = frames_landmarks[mit_hand[0]:mit_hand[-1] + 1]

    bilder = []
    spur = []
    for punkte in frames_landmarks:
        if punkte:
            spur.append(punkte[FINGER_IDX])
        bilder.append(male_frame(punkte, spur, f"Aufnahme: {pfad.name}"))

    # Am Ende klassifizieren und das Ergebnis ein paar Frames stehen lassen.
    label, score = klassifiziere(spur, modell)
    for _ in range(20):
        bilder.append(male_frame(None, spur,
                                 f"Aufnahme: {pfad.name}   Score: {score:.2f}",
                                 text_gross=f"Erkannt: {label}"))

    ziel.parent.mkdir(parents=True, exist_ok=True)
    # GIF speichern: 40 ms pro Bild = fluessig, aber kleine Datei.
    bilder[0].save(ziel, save_all=True, append_images=bilder[1:],
                   duration=40, loop=0)
    print(f"  {ziel} geschrieben ({len(bilder)} Frames, erkannt: {label})")
    return label


def main():
    parser = argparse.ArgumentParser("Demo-GIFs der Gesten-Klassifikation erzeugen")
    parser.add_argument("--letters", default="A,M,W",
                        help="Welche Buchstaben (Komma-Liste, Standard: A,M,W)")
    parser.add_argument("--out", default="images",
                        help="Zielordner fuer die GIFs (Standard: images)")
    args = parser.parse_args()

    modell_pfad = Path("data/hmm.pkl")
    if not modell_pfad.exists():
        print("Kein Modell gefunden. Bitte zuerst trainieren:  python train.py")
        return
    modell = HMMClassifier.load(modell_pfad)

    for buchstabe in [b.strip().upper() for b in args.letters.split(",") if b.strip()]:
        ordner = Path("recordings") / buchstabe
        # Aufnahmen mit Personen-Namen nehmen (die sind geprueft).
        kandidaten = sorted(ordner.glob(f"{buchstabe}-*-*.pkl")) or sorted(ordner.glob("*.pkl"))
        if not kandidaten:
            print(f"  {buchstabe}: keine Aufnahme gefunden, uebersprungen")
            continue
        # Fuer die Demo suchen wir ein Beispiel, das das Modell korrekt erkennt
        # (bei ~90% Accuracy gibt es auch Fehlgriffe -- die zeigen wir separat
        # in der Confusion Matrix, nicht im Demo-GIF).
        ziel = Path(args.out) / f"demo_{buchstabe}.gif"
        for kandidat in kandidaten[:10]:
            if gif_fuer_aufnahme(kandidat, modell, ziel) == buchstabe:
                break


if __name__ == "__main__":
    main()
