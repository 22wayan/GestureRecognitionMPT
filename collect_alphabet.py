import argparse

from GestureRecognition.labeling import collect_alphabet, _parse_letters


def main():
    parser = argparse.ArgumentParser(
        "Alphabet-Aufnahme (A-Z oder frei gewaehlte Buchstaben, je mehrfach pro Person)"
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
    parser.add_argument(
        "--start",
        help="Erster Buchstabe eines zusammenhaengenden Bereichs (Standard: A)",
    )
    parser.add_argument(
        "--end",
        help="Letzter Buchstabe eines zusammenhaengenden Bereichs (Standard: Z)",
    )
    args = parser.parse_args()

    person = args.person or input("Dein Name / Kuerzel: ").strip()

    # Buchstaben-Auswahl:
    #  - Mit --start/--end bleibt es beim bisherigen Bereichs-Modus.
    #  - Sonst fragen wir interaktiv nach frei gewaehlten Buchstaben.
    #    Leere Eingabe = ganzes Alphabet A-Z.
    if args.start or args.end:
        collect_alphabet(person, times=args.times, start=args.start, end=args.end)
    else:
        while True:
            raw = input(
                "Welche Buchstaben aufnehmen? "
                "(leer = alle A-Z; Tipp fuer schwache: Q,C,W,O,M): "
            ).strip()
            if not raw:
                letters = None  # None -> collect_alphabet nimmt A-Z
                break
            letters = _parse_letters(raw)
            if letters:
                break
            print("Keine gueltigen Buchstaben (A-Z) erkannt. Bitte nochmal eingeben.")
        collect_alphabet(person, times=args.times, letters=letters)


if __name__ == "__main__":
    main()
