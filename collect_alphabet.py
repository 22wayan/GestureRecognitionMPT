import argparse

from GestureRecognition.labeling import collect_alphabet


def main():
    parser = argparse.ArgumentParser(
        "Alphabet-Aufnahme (A-Z, je 1x pro Person)"
    )
    parser.add_argument(
        "--person",
        help="Dein Name / Kuerzel (wird sonst interaktiv abgefragt)",
    )
    args = parser.parse_args()

    person = args.person or input("Dein Name / Kuerzel: ").strip()
    collect_alphabet(person)


if __name__ == "__main__":
    main()
