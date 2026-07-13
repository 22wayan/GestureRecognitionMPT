"""
Schnell-Recorder fuer die Alphabet-Aufnahme.

Ein einziges Kamera-Fenster, in dem man alle Takes hintereinander aufnimmt --
kein staendiges Fenster-Auf/Zu wie beim SignalHub-Ablauf. Der gezeichnete Strich
BLEIBT waehrend eines Takes stehen (bis gespeichert/verworfen).

Steuerung (im Kamera-Fenster):
  LEERTASTE  -> Aufnahme starten; nochmal druecken -> Take speichern
  R          -> aktuellen Take verwerfen / Strich loeschen
  N          -> diesen Buchstaben ueberspringen
  Q / ESC    -> beenden

Die Dateien landen als recordings/<Buchstabe>/<Buchstabe>-<person>-<n>.pkl im
exakt gleichen Format, das clean_recordings()/dataset_building() lesen -- also
direkt kompatibel mit den bestehenden Aufnahmen und dem Training.
"""
import sys
import pickle
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

HERE = Path(__file__).resolve().parent
MODEL = HERE / "hand_landmarker.task"
REC_DIR = HERE / "recordings"
ALPHABET = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
TIMES = 15
FINGER_IDX = 8          # Zeigefingerspitze
MIN_FRAMES = 20         # weniger -> Take zu kurz, wird nicht gespeichert
FLIP = True             # wie config.yml (webcam.flip: True)


def sanitize(name: str) -> str:
    for c in " /:\\":
        name = name.replace(c, "")
    return name


def canonical_person(person: str) -> str:
    # Falls es schon Aufnahmen dieser Person gibt (egal welche Schreibweise),
    # deren exakte Schreibweise uebernehmen -> einheitliche Dateinamen.
    pl = person.lower()
    if REC_DIR.exists():
        for d in sorted(REC_DIR.glob("*")):
            for f in d.glob("*.pkl"):
                parts = f.stem.split("-")
                if len(parts) == 3 and parts[1].lower() == pl:
                    return parts[1]
    return person


def person_takes(letter: str, person: str) -> int:
    # Groß-/Kleinschreibung des Namens ignorieren, damit "azad" auch die
    # bereits vorhandenen "Azad"-Aufnahmen findet (Resume funktioniert sonst nicht).
    d = REC_DIR / letter
    if not d.exists():
        return 0
    pl = person.lower()
    cnt = 0
    for f in d.glob(f"{letter}-*.pkl"):
        parts = f.stem.split("-")
        if len(parts) == 3 and parts[1].lower() == pl:
            cnt += 1
    return cnt


def newest_person_file(person: str):
    # Die zuletzt gespeicherte Aufnahme dieser Person finden (fuer "Rueckgaengig").
    pl = person.lower()
    newest, newest_t = None, -1.0
    if REC_DIR.exists():
        for d in REC_DIR.glob("*"):
            for f in d.glob("*.pkl"):
                parts = f.stem.split("-")
                if len(parts) == 3 and parts[1].lower() == pl:
                    t = f.stat().st_mtime
                    if t > newest_t:
                        newest_t, newest = t, f
    return newest


def first_incomplete_index(person, symbols, times):
    # Sucht das erste Zeichen, von dem noch nicht genug Aufnahmen da sind.
    for i, L in enumerate(symbols):
        if person_takes(L, person) < times:
            return i
    return len(symbols)


def next_take_path(letter: str, person: str) -> Path:
    d = REC_DIR / letter
    d.mkdir(parents=True, exist_ok=True)
    n = 1
    while (d / f"{letter}-{person}-{n}.pkl").exists():
        n += 1
    return d / f"{letter}-{person}-{n}.pkl"


def make_detector():
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(opts)


def put_lines(frame, lines, org=(24, 46), scale=0.9, dy=40):
    x, y = org
    for i, (text, color) in enumerate(lines):
        pos = (x, y + i * dy)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def main():
    # Kommandozeile lesen: Name, welche Zeichen, wie viele Aufnahmen.
    import argparse
    parser = argparse.ArgumentParser("Schnell-Aufnahme (A-Z oder eigene neue Symbole)")
    parser.add_argument("person", nargs="?", help="Dein Name / Kuerzel")
    parser.add_argument(
        "--symbols",
        help="Eigene neue Zeichen statt A-Z, mit Komma getrennt (z.B. STERN,HERZ,BLITZ). "
             "Ohne Angabe wird das ganze Alphabet A-Z aufgenommen.",
    )
    parser.add_argument("--times", type=int, default=TIMES,
                        help=f"Wie viele Aufnahmen pro Zeichen (Standard {TIMES}).")
    args = parser.parse_args()

    person = sanitize(args.person if args.person else input("Dein Name / Kuerzel: ").strip())
    if not person:
        print("Kein Name angegeben.")
        return
    person = canonical_person(person)   # vorhandene Schreibweise uebernehmen (z.B. "Azad")

    # Welche Zeichen sollen aufgenommen werden?
    # Mit --symbols: eigene neue Zeichen. Ohne: das ganze Alphabet A-Z.
    if args.symbols:
        symbols = [sanitize(s) for s in args.symbols.split(",") if sanitize(s)]
    else:
        symbols = list(ALPHABET)
    times = args.times
    print(f"Aufnahme fuer: {person}  |  Zeichen: {', '.join(symbols)}  |  je {times}x")

    detector = make_detector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera nicht verfuegbar (Berechtigung?).")
        return
    for _ in range(8):        # warmup
        cap.read()

    win = "Schnell-Aufnahme"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1120, 630)

    recording = False
    frames = []               # aufgenommene Detector-Frames des aktuellen Takes
    trail = []                # (px) Punkte des Zeigefingers -> bleibender Strich
    hand_frames = 0
    flash, flash_msg, flash_col = 0, "", (255, 255, 255)
    saved_total = 0

    li = 0
    while li < len(symbols):
        letter = symbols[li]
        have = person_takes(letter, person)
        if have >= times and not recording:
            li += 1
            continue

        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if FLIP:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        hands = getattr(result, "hand_landmarks", []) or []

        det_signal = {"hands": []}
        has_hand = bool(hands)
        if has_hand:
            hand = hands[0]
            det_signal["hands"] = [
                {"id": 0, "landmarks": [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand]}
            ]
            # Handpunkte einzeichnen (gruen)
            for lm in hand:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)
            if recording:
                tip = hand[FINGER_IDX]
                trail.append((int(tip.x * w), int(tip.y * h)))

        if recording:
            frames.append({"detector": det_signal, "galy": None})
            if has_hand:
                hand_frames += 1

        # Bleibender Strich (blau/orange), dick
        for i in range(1, len(trail)):
            cv2.line(frame, trail[i - 1], trail[i], (255, 120, 0), 5, cv2.LINE_AA)

        # HUD
        if recording:
            status = ("● AUFNAHME laeuft   —   LEERTASTE = SPEICHERN", (60, 60, 255))
            cv2.circle(frame, (w - 45, 45), 16, (0, 0, 255), -1)
        else:
            status = ("LEERTASTE = Aufnahme starten", (80, 220, 80))
        lines = [
            (f"Zeichen {letter}    Take {min(have + 1, times)}/{times}", (255, 255, 255)),
            status,
            ("R = nochmal   BACKSPACE = letzten loeschen   N = ueberspr.   Q = Ende", (220, 220, 220)),
        ]
        if flash > 0:
            lines.append((flash_msg, flash_col))
            flash -= 1
        put_lines(frame, lines)

        cv2.imshow(win, frame)

        # Fenster ueber X geschlossen? -> beenden
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            if not recording:
                recording, frames, trail, hand_frames = True, [], [], 0
            else:
                recording = False
                if hand_frames >= MIN_FRAMES:
                    path = next_take_path(letter, person)
                    with open(path, "wb") as f:
                        pickle.dump(
                            {"detector": frames, "trailmarker": [],
                             "preprocessor": [], "hiddenmarkov": []}, f)
                    saved_total += 1
                    flash, flash_msg, flash_col = 25, f"gespeichert  ({path.name})", (80, 220, 80)
                    if person_takes(letter, person) >= times:
                        li += 1     # Zeichen fertig -> naechstes
                else:
                    flash, flash_msg, flash_col = 25, "zu kurz / keine Hand - nochmal", (60, 60, 255)
                frames, trail, hand_frames = [], [], 0
        elif key == ord("r"):
            recording, frames, trail, hand_frames = False, [], [], 0
            flash, flash_msg, flash_col = 15, "verworfen", (60, 60, 255)
        elif key == ord("n"):
            recording, frames, trail, hand_frames = False, [], [], 0
            li += 1
        elif key in (8, 127):    # Backspace / Entf  -> letzten gespeicherten Take loeschen
            recording, frames, trail, hand_frames = False, [], [], 0
            victim = newest_person_file(person)
            if victim is not None:
                nm = victim.name
                victim.unlink()
                li = first_incomplete_index(person, symbols, times)   # zum ersten offenen Zeichen
                flash, flash_msg, flash_col = 30, f"geloescht: {nm}", (0, 200, 255)
            else:
                flash, flash_msg, flash_col = 20, "nichts zum Loeschen", (60, 60, 255)

    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    print(f"Fertig. Diese Sitzung gespeichert: {saved_total} Take(s).")


if __name__ == "__main__":
    main()
