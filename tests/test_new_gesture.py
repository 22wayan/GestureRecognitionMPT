"""
Test: Vorbereitung auf eine LIVE vom Pruefer aufgenommene NEUE Geste (Aufgabe 2).

Hintergrund
-----------
Die Aufgabenstellung verlangt ausdruecklich (GestureRecognitionMPT.md, Aufgabe 2):

    "In der Pruefung werden neue Daten live vom Pruefer aufgenommen.
     Ihr System muss darauf vorbereitet sein."

Der Pruefer kann also eine voellig NEUE Geste aufnehmen -- nicht nur A-Z. Dieser
Test beweist headless (ohne Webcam), dass der komplette Loop
``Aufnahmen -> dataset_building -> HMMClassifier.fit -> predict`` mit einem
BELIEBIGEN, vorher nie gesehenen Gesten-Label funktioniert.

Vorgehen
--------
Es wird ein temporaeres ``recordings/`` mit zwei frei erfundenen Gesten-Namen
("WELLE", "KREIS") angelegt. Als Rohdaten dienen echte, formverschiedene
Aufnahmen (L- bzw. O-Bewegung) -- entscheidend ist NICHT der Buchstabe, sondern
dass das System ein neues Label durchgaengig lernt und wiedererkennt. Genau das
passiert in der Pruefung, wenn der Pruefer eine neue Geste einspielt.

Die eigentliche Webcam-Aufnahme selbst ist ein manueller Schritt
(``python collect_alphabet.py`` / ``schnell_aufnahme.py`` / ``data_labeling``);
dieser Test sichert den datengetriebenen Teil dahinter ab.

Laeuft unter pytest und standalone:

    python tests/test_new_gesture.py
"""

import glob
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from GestureRecognition.labeling import dataset_building  # noqa: E402
from GestureRecognition.hmmclassifier import HMMClassifier  # noqa: E402

# Zwei frei erfundene NEUE Gesten (kein A-Z) und die formverschiedenen Buchstaben,
# aus denen wir echte Rohaufnahmen als Stellvertreter ziehen.
NEUE_GESTEN = {"WELLE": "L", "KREIS": "O"}
SAMPLES_PRO_GESTE = 14
TEST_PERSON = "yannik"


def _baue_temp_recordings(basis: Path) -> Path:
    """Legt ein temporaeres recordings/<NEUE_GESTE>/ mit echten Aufnahmen an."""
    rec = basis / "recordings"
    for neue_geste, quell_buchstabe in NEUE_GESTEN.items():
        ziel = rec / neue_geste
        ziel.mkdir(parents=True)
        quellen = sorted(
            glob.glob(str(ROOT / f"recordings/{quell_buchstabe}/{quell_buchstabe}-{TEST_PERSON}-*.pkl"))
        )[:SAMPLES_PRO_GESTE]
        assert quellen, f"Keine Quellaufnahmen fuer '{quell_buchstabe}-{TEST_PERSON}-*'."
        for index, quelle in enumerate(quellen):
            shutil.copy(quelle, ziel / f"{neue_geste}-{index}.pkl")
    return rec


def test_neue_geste_wird_end_to_end_gelernt_und_erkannt():
    """Ein beliebiges neues Gesten-Label muss durch die ganze Pipeline laufen und
    wiedererkannt werden -- der Nachweis fuer 'auf neue Live-Gesten vorbereitet'.
    """
    # hmmlearn-Konvergenz-Warnungen bei kleinen Sets stummschalten (kein Fehler).
    logging.disable(logging.INFO)
    tmp = Path(tempfile.mkdtemp())
    try:
        rec = _baue_temp_recordings(tmp)

        data = dataset_building(
            tmp / "ds.pkl", recordings_dir=rec, test_size=0.3, random_state=0
        )

        # 1) Die neuen Labels muessen unveraendert durch den Datensatz-Bau kommen.
        assert sorted(data["classes"]) == sorted(NEUE_GESTEN), (
            f"Neue Gesten-Labels nicht durchgereicht: {data['classes']}"
        )
        assert len(data["lengths_test"]) > 0, "Kein Test-Split fuer die neuen Gesten."

        # 2) Der Classifier muss die neuen Labels als Klassen lernen.
        clf = HMMClassifier(n_components=4, random_state=0)
        clf.fit(data["X_train"], data["y_train"], data["lengths_train"])
        assert sorted(str(c) for c in clf.classes_) == sorted(NEUE_GESTEN), (
            f"Neue Gesten nicht als Klassen gelernt: {clf.classes_}"
        )

        # 3) Held-out-Sequenzen der neuen Gesten muessen wiedererkannt werden.
        predictions = [str(p) for p in clf.predict(data["X_test"], data["lengths_test"])]
        truth = [str(t) for t in data["y_test"]]
        accuracy = np.mean([p == t for p, t in zip(predictions, truth)])
        assert accuracy >= 0.7, (
            f"Neue Gesten zu schlecht erkannt: {accuracy:.3f}\n"
            f"pred={predictions}\ntruth={truth}"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        logging.disable(logging.NOTSET)


if __name__ == "__main__":
    test_neue_geste_wird_end_to_end_gelernt_und_erkannt()
    print("OK: System lernt und erkennt eine beliebige neue Geste end-to-end.")
