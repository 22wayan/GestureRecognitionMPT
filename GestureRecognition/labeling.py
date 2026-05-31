import os
import sys
import shutil
import tempfile
import subprocess


def _next_recording_index(label_dir: str) -> int:
    """
    Sucht die naechste freie Nummer fuer eine Aufnahme.

    Wir starten bei 1 und zaehlen hoch, solange die Datei schon existiert.
    So wird nie eine bestehende Aufnahme ueberschrieben.
    """
    i = 1
    while os.path.exists(os.path.join(label_dir, f"recording_{i}.pkl")):
        i += 1
    return i


def data_labeling(times: int, label: str):
    """
    Aufnahme-Workflow fuer eigene Trainingsdaten.

    Es werden nacheinander Aufnahmen gemacht. Pro Aufnahme entscheidet der
    Benutzer, ob die Aufnahme gespeichert, verworfen oder der ganze Vorgang
    abgebrochen wird. Gespeicherte Aufnahmen landen unter
    ``data/{label}/recording_{i}.pkl``.

    Parameters
    ----------
    times : int
        Wie viele Aufnahmen gespeichert werden sollen.
    label : str
        Name der Geste / Klasse (z. B. "A").
    """

    # Zielordner fuer dieses Label, z. B. data/A
    label_dir = os.path.join("data", label)
    # Ordner anlegen, falls er noch nicht existiert (kein Fehler, wenn schon da)
    os.makedirs(label_dir, exist_ok=True)

    # Kurze Anleitung fuer den Benutzer ausgeben
    print("=" * 50)
    print(f"Aufnahme-Workflow fuer Label: {label}")
    print(f"Es sollen {times} Aufnahme(n) gespeichert werden.")
    print("Ablauf pro Aufnahme:")
    print("  1. Ein Fenster oeffnet sich und nimmt die Geste auf.")
    print("  2. Fenster schliessen, um die Aufnahme zu beenden.")
    print("  3. Danach entscheiden: speichern / verwerfen / abbrechen.")
    print("=" * 50)

    # Zaehlt, wie viele Aufnahmen schon erfolgreich gespeichert wurden
    saved = 0

    # Schleife laeuft, bis die gewuenschte Anzahl gespeichert wurde
    while saved < times:
        print()
        print(f"--- Aufnahme {saved + 1} von {times} (Label: {label}) ---")
        input("Druecke ENTER, um die Aufnahme zu starten ...")

        # Temporaere Datei fuer diese eine Aufnahme erstellen.
        # SignalHub schreibt die Aufnahme erst beim sauberen Beenden in diese Datei.
        # Wir loeschen die leere Temp-Datei wieder, damit ihr blosses Vorhandensein
        # spaeter bedeutet: "Aufnahme war erfolgreich".
        fd, temp_file = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)
        os.remove(temp_file)

        try:
            # main.py als Subprocess starten und die Aufnahme in die Temp-Datei lenken.
            # --mode record sorgt dafuer, dass aufgenommen wird.
            cmd = [
                sys.executable,
                "main.py",
                "--mode",
                "record",
                "--recorder.file",
                temp_file,
            ]
            subprocess.run(cmd)

            # Crash-Schutz: Wenn keine Datei entstanden ist, ist beim Aufnehmen
            # etwas schiefgelaufen. Wir speichern dann nichts und versuchen es erneut.
            if not os.path.exists(temp_file):
                print("Es wurde keine Aufnahme erzeugt. Bitte erneut versuchen.")
                continue

            # Benutzer entscheiden lassen, was mit der Aufnahme passiert
            print("Was soll mit der Aufnahme passieren?")
            print("  [s] speichern")
            print("  [v] verwerfen")
            print("  [a] abbrechen")
            choice = input("Deine Wahl: ").strip().lower()

            if choice == "s":
                # Naechste freie Nummer suchen, damit nichts ueberschrieben wird
                index = _next_recording_index(label_dir)
                dest = os.path.join(label_dir, f"recording_{index}.pkl")
                # Temp-Datei an den endgueltigen Ort verschieben (erst jetzt "echte" Daten)
                shutil.move(temp_file, dest)
                saved += 1
                print(f"Gespeichert: {dest}")
            elif choice == "a":
                # Abbrechen: Temp-Datei loeschen und Schleife verlassen
                print("Abgebrochen.")
                break
            else:
                # Alles andere = verwerfen: Temp-Datei loeschen, gleiche Aufnahme wiederholen
                print("Aufnahme verworfen.")
        finally:
            # Aufraeumen: eine evtl. noch vorhandene Temp-Datei loeschen.
            # Das verhindert, dass verworfene oder abgebrochene Aufnahmen Reste hinterlassen.
            if os.path.exists(temp_file):
                os.remove(temp_file)

    # Abschluss-Meldung
    print()
    print(f"Fertig. {saved} Aufnahme(n) fuer Label '{label}' gespeichert.")




def dataset_building(output_path):
    """
    TODO: dataset_building: Trainingsdatensatz aus aufgenommenen Gesten erstellen

    Ziel:
    -----
    Implementiere eine Funktion, die alle aufgenommenen Daten lädt,
    verarbeitet und in eine Form bringt, die von eurem
    Hidden-Markov-Modell (HMM) Classifier verwendet werden kann.

    Anforderungen / Ideen:
    ----------------------

    1. Daten laden

       - Durchsuche deinen Trainingsdaten-Ordner
       - Organisiere Daten nach Labels

    2. Feature-Extraktion / Preprocessing

       - Überlege:
         - Welche Features braucht dein Modell?
         - Wie transformierst du die Rohdaten sinnvoll?
       - Wende eine konsistente Verarbeitung auf alle Sequenzen an

    3. Umgang mit Sequenzen

       - Daten sind zeitliche Sequenzen
       - Achte auf:
         - Unterschiedliche Längen
         - Konsistente Struktur

    4. Validierung

       - Entferne unbrauchbare Daten
         (z. B. zu kurze oder fehlerhafte Sequenzen)

    5. Ausgabeformat

       - Baue den Datensatz so, dass dein HMM direkt damit arbeiten kann
       - Das Format sollst du selbst definieren

    .. note::

       Es gibt hier keine vorgegebene „richtige“ Lösung.
       Wichtig ist, dass dein Datensatz konsistent und nutzbar ist.

    .. tip::

       Denke wie ein System-Designer:
       Wie müssen Daten aussehen, damit Training und Inferenz sauber funktionieren?

    .. warning::

       Inkonsistente Datenstrukturen sind eine der häufigsten Fehlerquellen
       beim Training von Sequenzmodellen.

    Erweiterung (optional):
    -----------------------

    - Normalisierung der Daten
    - Datenaugmentation
    - Debug-Ausgaben oder Visualisierung

    Parameters
    ----------
    output_path : Path or str
        Zielpfad für den erzeugten Trainingsdatensatz.
    """
    pass