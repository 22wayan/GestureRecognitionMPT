import argparse

from GestureRecognition.labeling import collect_alphabet


def main():
    parser = argparse.ArgumentParser(
        "Alphabet-Aufnahme (A-Z, je mehrfach pro Person)"
    )
    parser.add_argument(
        "--person",
        help="Dein Name / Kuerzel (wird sonst interaktiv abgefragt)",
    )
    parser.add_argument(
        "--times",
        type=int,
        default=15,
        help="Wie viele Takes pro Buchstabe (Standard: 15)",
    )
    args = parser.parse_args()

    person = args.person or input("Dein Name / Kuerzel: ").strip()
    collect_alphabet(person, times=args.times)


if __name__ == "__main__":
    main()
