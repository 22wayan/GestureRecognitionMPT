"""
Trainiert den HMM-Klassifikator auf den aufgenommenen Gesten und speichert ihn.

Ablauf (siehe Doku "Fit/Train HMM Classifier")
----------------------------------------------
1. ``dataset_building()`` -- baut aus ``recordings/`` einen Datensatz im Format
   ``(x, y, velocity)`` und teilt ihn stratifiziert in Train/Test (sequenzweise,
   also ohne Data-Leakage).
2. ``HMMClassifier.fit()`` -- trainiert ein eigenes HMM pro Klasse auf dem
   Train-Split.
3. Auswertung auf dem Test-Split (Accuracy) -- Test ist sauber vom Training
   getrennt, sonst waeren die Zahlen nicht aussagekraeftig.
4. ``.save()`` -- legt das Modell unter ``data/hmm.pkl`` ab, also genau dort,
   wo das :class:`HMMModule` es im Live-/Replay-Betrieb per Default laedt.

Das Trainingsformat ``(x, y, velocity)`` ist identisch zu dem, was der
Preprocessor live erzeugt -- nur so passt Training zu Inferenz.

Benutzung
---------
    python train.py                    # Standard: recordings/ -> data/hmm.pkl
    python train.py --grid-search      # bestes n_components per Cross-Validation
    python train.py --n-components 5   # feste Zustandszahl vorgeben
"""

import argparse
import logging

from GestureRecognition.hmmclassifier import HMMClassifier
from GestureRecognition.labeling import dataset_building

logger = logging.getLogger(__name__)


def _split_sequences(X, lengths):
    """Zerlegt ein konkateniertes ``(total_frames, n_features)``-Array anhand von
    ``lengths`` wieder in eine Liste einzelner Sequenzen.

    Die Grid-Search arbeitet auf einzelnen Sequenzen, ``dataset_building`` liefert
    aber ein zusammengehaengtes Array plus Laengenliste. Diese Hilfsfunktion baut
    die Sequenzen daraus zurueck.
    """
    sequences = []
    offset = 0
    for length in lengths:
        sequences.append(X[offset:offset + length])
        offset += length
    return sequences


def _accuracy(y_true, y_pred):
    """Anteil korrekter Vorhersagen (einfache Klassifikationsgenauigkeit)."""
    y_true = list(y_true)
    if not y_true:
        return 0.0
    correct = sum(true == pred for true, pred in zip(y_true, y_pred))
    return correct / len(y_true)


def train(
    recordings_dir="recordings",
    model_path="data/hmm.pkl",
    dataset_path="data/dataset.pkl",
    n_components=4,
    use_grid_search=False,
    test_size=0.2,
    random_state=42,
):
    """Baut den Datensatz, trainiert den Klassifikator und speichert ihn.

    Returns
    -------
    HMMClassifier
        Das trainierte Modell (zusaetzlich unter ``model_path`` gespeichert).
    """
    # 1) Datensatz bauen: bereinigen, Features extrahieren, Train/Test-Split.
    logger.info("Baue Datensatz aus '%s' ...", recordings_dir)
    data = dataset_building(
        dataset_path,
        recordings_dir=recordings_dir,
        test_size=test_size,
        random_state=random_state,
    )
    X_train, y_train, lengths_train = data["X_train"], data["y_train"], data["lengths_train"]
    X_test, y_test, lengths_test = data["X_test"], data["y_test"], data["lengths_test"]
    classes = data["classes"]
    logger.info(
        "Klassen (%d): %s | Train-Sequenzen: %d | Test-Sequenzen: %d",
        len(classes), classes, len(lengths_train), len(lengths_test),
    )

    # 2) Optional: bestes n_components per Cross-Validation auf dem Train-Split
    #    bestimmen. Der Import steht bewusst hier, damit die (nur dafuer noetige)
    #    matplotlib-Abhaengigkeit den Standardlauf nicht belastet.
    if use_grid_search:
        from GestureRecognition.grid_search import grid_search_n_components

        logger.info("Grid-Search fuer n_components (Cross-Validation) ...")
        train_sequences = _split_sequences(X_train, lengths_train)
        _, best = grid_search_n_components(train_sequences, list(y_train))
        n_components = best["n_components"]
        logger.info(
            "Bestes n_components = %d (CV-Accuracy %.3f)",
            n_components, best["mean_accuracy"],
        )

    # 3) Finales Modell auf dem kompletten Train-Split trainieren.
    logger.info("Trainiere HMMClassifier (n_components=%d) ...", n_components)
    classifier = HMMClassifier(n_components=n_components, random_state=random_state)
    classifier.fit(X_train, y_train, lengths_train)

    # 4) Auf dem Test-Split auswerten -- Test ist getrennt vom Training.
    if len(lengths_test) > 0:
        predictions = classifier.predict(X_test, lengths_test)
        accuracy = _accuracy(y_test, predictions)
        logger.info(
            "Test-Accuracy: %.3f  (%d Test-Sequenzen)", accuracy, len(lengths_test)
        )
    else:
        logger.warning("Kein Test-Split vorhanden -- Accuracy nicht berechenbar.")

    # 5) Modell speichern -- genau dorthin, wo das HMMModule es per Default laedt.
    classifier.save(model_path)
    logger.info("Fertig. Modell gespeichert unter '%s'.", model_path)
    return classifier


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="HMM-Klassifikator trainieren und als data/hmm.pkl speichern"
    )
    parser.add_argument(
        "--recordings-dir", default="recordings",
        help="Ordner mit den Aufnahmen (Standard: recordings)",
    )
    parser.add_argument(
        "--model-out", default="data/hmm.pkl",
        help="Zielpfad fuer das Modell (Standard: data/hmm.pkl)",
    )
    parser.add_argument(
        "--dataset-out", default="data/dataset.pkl",
        help="Zielpfad fuer den erzeugten Datensatz (Standard: data/dataset.pkl)",
    )
    parser.add_argument(
        "--n-components", type=int, default=4,
        help="Anzahl Zustaende pro HMM (Standard: 4)",
    )
    parser.add_argument(
        "--grid-search", action="store_true",
        help="bestes n_components per Cross-Validation suchen (ueberschreibt --n-components)",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Anteil des Test-Splits (Standard: 0.2)",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train(
        recordings_dir=args.recordings_dir,
        model_path=args.model_out,
        dataset_path=args.dataset_out,
        n_components=args.n_components,
        use_grid_search=args.grid_search,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
