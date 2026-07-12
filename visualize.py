"""
Erzeugt alle geforderten Visualisierungen und Metriken mit einem Befehl.

Deckt die drei Bausteine aus der Kriterien-Seite "Visualisierung ihres
Datensatzes" ab:

1. ``visualize_dataset()``   -- Datensatz visuell inspizieren
   (Trajektorien / Sequenzlaengen / Geschwindigkeitsprofile je Klasse).
2. ``replay_recordings()``   -- jede einzelne Rohaufnahme als Kachel; schlechte
   Aufnahmen (zu kurz / Tracking-Sprung / keine Hand) rot markiert (Replay).
3. ``evaluate_classifier()`` -- Accuracy (Standard + Neue Person) und
   Confusion Matrix auf getrennten Testdaten.

Benutzung
---------
    python visualize.py                 # alles, Standardpfade (recordings/ -> plots/)
    python visualize.py --skip-eval     # ohne (langsames) Training/Auswertung
    python visualize.py --recordings recordings --output plots
"""

import argparse
import logging

from GestureRecognition.visualization import (
    evaluate_classifier,
    replay_recordings,
    visualize_dataset,
)


def main():
    parser = argparse.ArgumentParser("Visualisierung & Exploration des Datensatzes")
    parser.add_argument("--recordings", default="recordings", help="Verzeichnis mit den Aufnahmen")
    parser.add_argument("--output", default="plots", help="Zielverzeichnis fuer die Plots")
    parser.add_argument(
        "--held-out-person",
        default="yannik",
        help="Person, die fuer die Neue-Person-Bewertung zurueckgehalten wird",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Die (langsame) Modell-Auswertung ueberspringen",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # 1) Datensatz-Uebersicht: alle Aufnahmen einer Klasse ueberlagert.
    print("\n=== 1/3  visualize_dataset: Datensatz-Uebersicht ===")
    dataset = visualize_dataset(args.recordings, output_dir=args.output)
    n_aufnahmen = sum(len(v) for v in dataset.values())
    print(f"  {len(dataset)} Klassen, {n_aufnahmen} saubere Aufnahmen -> {args.output}/")

    # 2) Replay-Galerie: jede Aufnahme einzeln, schlechte rot markiert.
    print("\n=== 2/3  replay_recordings: Aufnahme-Galerie ===")
    galerie = replay_recordings(args.recordings, output_dir=f"{args.output}/gallery")
    gesamt = sum(g["gesamt"] for g in galerie.values())
    markiert = sum(g["markiert"] for g in galerie.values())
    print(f"  {gesamt} Aufnahmen insgesamt, {markiert} markiert -> {args.output}/gallery/")
    # Die auffaelligsten Buchstaben (viele Markierungen) kurz auflisten.
    auffaellig = sorted(galerie.items(), key=lambda kv: kv[1]["markiert"], reverse=True)
    for label, g in auffaellig[:5]:
        if g["markiert"]:
            print(f"    {label}: {g['markiert']}/{g['gesamt']} markiert")

    # 3) Modell-Auswertung: Accuracy + Confusion Matrix (getrennte Testdaten).
    if args.skip_eval:
        print("\n=== 3/3  evaluate_classifier: uebersprungen (--skip-eval) ===")
        return

    print("\n=== 3/3  evaluate_classifier: Accuracy + Confusion Matrix ===")
    ergebnis = evaluate_classifier(
        args.recordings, output_dir=args.output, held_out_person=args.held_out_person
    )
    print(f"  Accuracy Standard (gleiche Personen):        {ergebnis['accuracy_standard']:.1%}")
    print(
        f"  Accuracy Neue Person ({ergebnis['held_out_person']}):"
        f"          {ergebnis['accuracy_new_person']:.1%}"
    )
    print(f"  Confusion Matrix -> {args.output}/confusion_matrix.png")


if __name__ == "__main__":
    main()
