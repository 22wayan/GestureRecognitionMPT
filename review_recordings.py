"""
Prueft die aufgenommenen Gesten und sortiert unbrauchbare aus.

Eine Aufnahme ist nur dann brauchbar, wenn sie spaeter auch wirklich im
Datensatz landet. Darum pruefen wir hier genau dasselbe wie ``dataset_building``:

  1. Lang genug?      -> mindestens ``min_length`` Frames (nach dem Trimmen
                         der Frames ohne Hand am Anfang/Ende)
  2. Keine Spruenge?  -> kein Sprung der Fingerspitze > ``max_jump`` von einem
                         Frame zum naechsten (typischer Tracking-Fehler)
  3. Keine Luecke?    -> kein NaN mitten in der Bewegung (Hand kurz verloren)

Es werden NUR die eigenen Aufnahmen einer Person angefasst, also Dateien wie
``A-yannik-1.pkl``. Die geteilten Team-Altdaten (``A-1773050612....pkl``)
bleiben unberuehrt.

Benutzung
---------
    python review_recordings.py --person yannik              # nur anschauen
    python review_recordings.py --person yannik --quarantine # schlechte verschieben
"""

import argparse
import pickle
import shutil
from pathlib import Path

import numpy as np

from GestureRecognition.labeling import _extract_trajectory, _is_outlier, ALPHABET


def biggest_jump(traj):
    """Gibt den groessten Abstand zwischen zwei aufeinanderfolgenden Punkten zurueck."""
    # Nur die Punkte nehmen, an denen die Hand erkannt wurde (keine NaN).
    valid = traj[~np.isnan(traj).any(axis=1)]
    groesster = 0.0
    for i in range(1, len(valid)):
        abstand = float(np.linalg.norm(valid[i] - valid[i - 1]))
        if abstand > groesster:
            groesster = abstand
    return groesster


def check_recording(path, finger_idx, min_length, max_jump):
    """
    Prueft eine einzelne Aufnahme.

    Gibt (ok, grund) zurueck:
      ok    -> True, wenn die Aufnahme brauchbar ist
      grund -> "OK" oder warum sie verworfen wird
    """
    with open(path, "rb") as f:
        recording = pickle.load(f)

    # Die Fingerspitzen-Bewegung aus der Aufnahme holen.
    traj = _extract_trajectory(recording, finger_idx)

    if traj is None:
        return False, "keine Hand erkannt"
    if len(traj) < min_length:
        return False, f"zu kurz ({len(traj)} < {min_length} Frames)"
    if _is_outlier(traj, max_jump):
        return False, f"Tracking-Sprung ({biggest_jump(traj):.3f})"
    if np.isnan(traj).any():
        return False, "NaN mittendrin (Hand kurz verloren)"
    return True, "OK"


def find_person_files(label_dir, letter, person):
    """Alle Aufnahme-Dateien einer Person fuer einen Buchstaben finden."""
    files = sorted(label_dir.glob(f"{letter}-{person}-*.pkl"))
    # Aeltere Aufnahmen hatten evtl. keine Nummer (z.B. A-yannik.pkl).
    legacy = label_dir / f"{letter}-{person}.pkl"
    if legacy.exists():
        files.append(legacy)
    return files


def move_to_quarantine(path, quarantine_dir, letter):
    """Verschiebt eine schlechte Aufnahme in den Quarantaene-Ordner (loescht nichts)."""
    ziel_ordner = quarantine_dir / letter
    ziel_ordner.mkdir(parents=True, exist_ok=True)
    ziel = ziel_ordner / path.name
    # Falls dort schon eine Datei mit dem Namen liegt: eine Nummer anhaengen.
    nummer = 1
    while ziel.exists():
        ziel = ziel_ordner / f"{path.stem}__{nummer}{path.suffix}"
        nummer += 1
    shutil.move(str(path), str(ziel))


def main():
    parser = argparse.ArgumentParser("Aufnahmen pruefen und schlechte aussortieren")
    parser.add_argument("--person", required=True, help="Name / Kuerzel (wie bei der Aufnahme)")
    parser.add_argument("--recordings-dir", default="recordings")
    parser.add_argument("--quarantine-dir", default="recordings_rejected")
    parser.add_argument("--finger-idx", type=int, default=8)
    parser.add_argument("--min-length", type=int, default=15)
    parser.add_argument("--max-jump", type=float, default=0.15)
    parser.add_argument("--target", type=int, default=15,
                        help="Gewuenschte Anzahl guter Aufnahmen pro Buchstabe")
    parser.add_argument("--quarantine", action="store_true",
                        help="Schlechte wirklich verschieben (ohne dieses Flag nur anzeigen)")
    args = parser.parse_args()

    recordings_dir = Path(args.recordings_dir)
    quarantine_dir = Path(args.quarantine_dir)
    person = args.person.strip()

    if args.quarantine:
        print(f"Pruefe Aufnahmen von '{person}' -- schlechte werden verschoben.")
    else:
        print(f"Pruefe Aufnahmen von '{person}' -- nur Anzeige (nichts wird veraendert).")
    print(f"{'Buchstabe':<10}{'gut':>5}{'gesamt':>8}{'fehlt':>7}   verworfen")
    print("-" * 60)

    summe_gut = 0
    summe_fehlt = 0

    # Jeden Buchstaben A-Z durchgehen.
    for letter in ALPHABET:
        label_dir = recordings_dir / letter
        if not label_dir.is_dir():
            continue

        files = find_person_files(label_dir, letter, person)
        if not files:
            continue  # diesen Buchstaben hat die Person noch nicht aufgenommen

        gute = 0
        verworfen = []
        for path in files:
            ok, grund = check_recording(path, args.finger_idx, args.min_length, args.max_jump)
            if ok:
                gute += 1
            else:
                verworfen.append((path, grund))
                if args.quarantine:
                    move_to_quarantine(path, quarantine_dir, letter)

        fehlt = max(0, args.target - gute)
        summe_gut += gute
        summe_fehlt += fehlt

        text = ", ".join(f"{p.name}: {g}" for p, g in verworfen) if verworfen else "-"
        print(f"{letter:<10}{gute:>5}{len(files):>8}{fehlt:>7}   {text}")

    print("-" * 60)
    print(f"Gute Aufnahmen gesamt: {summe_gut}")
    if summe_fehlt > 0:
        print(f"Noch nachzunehmen bis Ziel {args.target}: {summe_fehlt} Aufnahme(n).")
        print("=> einfach collect_alphabet.py erneut starten.")
    else:
        print(f"Alle aufgenommenen Buchstaben haben das Ziel von {args.target} erreicht.")


if __name__ == "__main__":
    main()
