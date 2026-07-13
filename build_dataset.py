"""
Baut aus den Rohaufnahmen einen Trainingsdatensatz und speichert ihn als Pickle.

Duenner CLI-Wrapper um :func:`GestureRecognition.labeling.dataset_building` --
damit der Datensatz-Bau (wie die anderen Workflows) auch direkt von der
Kommandozeile aufrufbar ist, nicht nur per Python-Import (Issue #28).

Benutzung
---------
    python build_dataset.py                         # recordings/ -> data/dataset.pkl
    python build_dataset.py -o data/mein.pkl        # anderer Zielpfad
    python build_dataset.py --min-length 20         # strengere Kuratierung
"""

import argparse
import logging

from GestureRecognition.labeling import dataset_building

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Trainingsdatensatz aus recordings/ bauen und als Pickle speichern"
    )
    parser.add_argument(
        "-o", "--output", default="data/dataset.pkl",
        help="Zielpfad fuer den Datensatz (Standard: data/dataset.pkl)",
    )
    parser.add_argument(
        "--recordings-dir", default="recordings",
        help="Ordner mit den Aufnahmen (Standard: recordings)",
    )
    parser.add_argument(
        "--finger-idx", type=int, default=8,
        help="MediaPipe-Landmark des Fingers (Standard: 8 = Zeigefingerspitze)",
    )
    parser.add_argument(
        "--min-length", type=int, default=15,
        help="Minimale Sequenzlaenge nach dem Trimmen (Standard: 15)",
    )
    parser.add_argument(
        "--max-jump", type=float, default=0.15,
        help="Maximal erlaubter Frame-zu-Frame-Sprung (Standard: 0.15)",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Anteil des Test-Splits (Standard: 0.2)",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data = dataset_building(
        args.output,
        recordings_dir=args.recordings_dir,
        finger_idx=args.finger_idx,
        min_length=args.min_length,
        max_jump=args.max_jump,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    logger.info(
        "Fertig. Klassen: %d | Train-Sequenzen: %d | Test-Sequenzen: %d | gespeichert: %s",
        len(data["classes"]), len(data["lengths_train"]), len(data["lengths_test"]),
        args.output,
    )


if __name__ == "__main__":
    main()
