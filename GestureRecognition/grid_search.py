"""
Grid Search: die beste Anzahl verborgener HMM-Zustaende (``n_components``) finden.

Idee in einem Satz
------------------
Statt die Zustandszahl zu raten, probieren wir mehrere Werte systematisch durch
und messen fair, welcher am besten verallgemeinert.

Die zwei Begriffe (fuer Einsteiger)
-----------------------------------
- **Grid Search**: mehrere Werte fuer einen Parameter der Reihe nach ausprobieren
  und jeden messen. Hier ist der Parameter ``n_components`` -- also wie viele
  "Bewegungsphasen" (verborgene Zustaende) jedes Buchstaben-Modell benutzen darf.
- **Cross-Validation (CV)**: die faire Art zu messen. Der Datensatz wird in
  ``n_splits`` gleich grosse Teile (Folds) zerlegt. Dann wird ``n_splits``-mal
  trainiert: jedes Mal ist EIN Teil Test und der Rest Training, und jeder Teil
  ist genau einmal der Test. Am Ende mittelt man die Genauigkeiten. So haengt das
  Ergebnis nicht vom Zufall einer einzigen Aufteilung ab. Die Streuung ueber die
  Folds verraet zusaetzlich, wie STABIL eine Konfiguration ist.

Was diese Datei bewusst NICHT (mehr) tut
----------------------------------------
Frueher schrieb dieses Modul zusaetzlich CSV-/Markdown-/JSON-Reports und einen
Plot und verglich auch ``diag`` gegen ``full``. Das wurde entfernt, um die Suche
einfach und erklaerbar zu halten:

- ``covariance_type="full"`` ist auf unseren Daten numerisch instabil (bricht bei
  mehr Zustaenden ab) und kaum genauer -> wir bleiben bei ``"diag"``. Details und
  Messwerte stehen in ``docs/grid-search.md``.
- Die Ergebnisse werden einfach ZURUECKGEGEBEN (und koennen ausgedruckt werden),
  statt in Dateien geschrieben. Das genuegt und ist leichter zu verstehen.
"""

from collections import defaultdict

import numpy as np

from GestureRecognition.hmmclassifier import HMMClassifier


def _stratified_folds(labels, n_splits, random_state):
    """
    Teilt die Aufnahmen in ``n_splits`` Gruppen (Folds) fuer die Cross-Validation
    -- und zwar STRATIFIZIERT: jede Gestenklasse ist in jedem Fold ungefaehr gleich
    stark vertreten.

    Warum selbst gebaut (statt einer Bibliothek)? Weil die Logik einfach und gut
    erklaerbar ist: Wir gehen Klasse fuer Klasse durch, mischen deren Aufnahmen und
    verteilen sie reihum ("round-robin") auf die Folds. So bekommt jeder Fold von
    jeder Klasse ungefaehr gleich viele Aufnahmen.
    """
    # Pro Klasse merken, welche Aufnahme-Positionen (Indizes) zu ihr gehoeren.
    per_class = defaultdict(list)
    for index, label in enumerate(labels):
        per_class[label].append(index)

    # Nicht mehr Folds bilden, als die kleinste Klasse Aufnahmen hat -- sonst
    # blieben Folds leer. Beispiel: kleinste Klasse hat 3 Aufnahmen -> max. 3 Folds.
    smallest_class = min(len(indices) for indices in per_class.values())
    n_splits = min(n_splits, smallest_class)
    if n_splits < 2:
        raise ValueError(
            "Fuer Cross-Validation braucht jede Klasse mindestens 2 Aufnahmen."
        )

    # Fester Zufalls-Generator -> gleicher Seed ergibt immer dieselben Folds
    # (wichtig, damit alle getesteten Zustandszahlen auf DENSELBEN Aufteilungen
    # verglichen werden -- nur so ist der Vergleich fair).
    rng = np.random.default_rng(random_state)
    folds = [[] for _ in range(n_splits)]

    # Jede Klasse getrennt mischen und reihum auf die Folds verteilen.
    for indices in per_class.values():
        rng.shuffle(indices)
        for position, sample_index in enumerate(indices):
            folds[position % n_splits].append(sample_index)

    return folds


def _accuracy(y_true, y_pred):
    """Anteil richtig vorhergesagter Gesten (0.0 bis 1.0)."""
    if not y_true:
        return 0.0
    richtig = sum(wahr == geraten for wahr, geraten in zip(y_true, y_pred))
    return richtig / len(y_true)


def _stack(sequences):
    """
    Haengt eine Liste von Sequenzen fuer den ``HMMClassifier`` aneinander.

    Der Klassifikator erwartet alle Sequenzen in EINEM langen Array plus eine Liste
    der Einzel-Laengen -- damit er weiss, wo eine Aufnahme aufhoert und die naechste
    anfaengt.
    """
    lengths = [len(np.asarray(s)) for s in sequences]
    X = np.vstack([np.asarray(s, dtype=float) for s in sequences])
    return X, lengths


def _cross_validate(sequences, labels, n_components, covariance_type, n_splits, random_state):
    """
    Bewertet EINE Konfiguration (eine feste Zustandszahl) per Cross-Validation.

    Fuer jeden Fold: Fold = Test, Rest = Training. Es wird ein FRISCHER Klassifikator
    trainiert und auf dem Test-Fold gemessen. Zurueck kommt die mittlere Genauigkeit
    und ihre Streuung (wie stark die Fold-Ergebnisse schwanken).
    """
    folds = _stratified_folds(labels, n_splits, random_state)
    fold_accuracies = []

    for test_indices in folds:
        test_set = set(test_indices)
        train_indices = [i for i in range(len(sequences)) if i not in test_set]

        # Trainings- und Test-Aufnahmen anhand der Indizes zusammenstellen.
        X_train, len_train = _stack([sequences[i] for i in train_indices])
        y_train = [labels[i] for i in train_indices]
        X_test, len_test = _stack([sequences[i] for i in test_indices])
        y_test = [labels[i] for i in test_indices]

        # Frischen Klassifikator mit dieser Zustandszahl trainieren und testen.
        clf = HMMClassifier(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
        )
        clf.fit(X_train, y_train, len_train)
        y_pred = clf.predict(X_test, len_test)
        fold_accuracies.append(_accuracy(y_test, y_pred))

    fold_accuracies = np.array(fold_accuracies)
    return {
        "n_components": int(n_components),
        "mean_accuracy": float(fold_accuracies.mean()),   # Durchschnitt ueber die Folds
        "std_accuracy": float(fold_accuracies.std()),     # Streuung = wie stabil
        "fold_accuracies": [round(float(a), 4) for a in fold_accuracies],
    }


def grid_search_n_components(
    sequences,
    labels,
    n_components_values=(4, 6, 8, 10, 12),
    covariance_type="diag",
    n_splits=5,
    random_state=42,
):
    """
    Probiert mehrere Zustandszahlen durch und findet die beste (Grid Search).

    Fuer jeden Wert in ``n_components_values`` wird per Cross-Validation die mittlere
    Genauigkeit gemessen. Zurueck kommen alle Ergebnisse und die beste Konfiguration.

    Parameters
    ----------
    sequences : list of ndarray
        Die Aufnahmen als ``(N, 3)``-Arrays (x, y, velocity), gleiche Reihenfolge
        wie ``labels``.
    labels : list of str
        Das Klassenlabel je Aufnahme.
    n_components_values : tuple of int
        Die zu testenden Zustandszahlen. Der Standard ``(4, 6, 8, 10, 12)`` deckt
        den ganzen Bereich ab: von "zu grob" (4) ueber den Sweet Spot bis
        "Overfitting" (12).
    covariance_type : str
        Form der Kovarianzmatrix. Standard ``"diag"`` (stabil). ``"full"`` ist
        numerisch instabil, siehe ``docs/grid-search.md``.
    n_splits : int
        Anzahl der Cross-Validation-Folds (Standard 5).
    random_state : int
        Zufalls-Seed fuer reproduzierbare Folds und Modelle.

    Returns
    -------
    (results, best) : tuple
        ``results`` -- Liste je getesteter Zustandszahl (nach ``n_components``
        sortiert), jeweils mit ``mean_accuracy``, ``std_accuracy`` und den
        einzelnen ``fold_accuracies``.
        ``best`` -- der Eintrag mit der hoechsten mittleren Genauigkeit (bei
        Gleichstand entscheidet die kleinere Streuung).

    Beispiel
    --------
    >>> results, best = grid_search_n_components(sequences, labels)
    >>> best["n_components"], round(best["mean_accuracy"], 3)
    (10, 0.908)
    """
    # NaN-Sicherung: Der HMM-Fit crasht bei nicht-endlichen Werten. Aufnahmen mit
    # NaN/inf hier ueberspringen. (Der Trainingspfad dataset_building filtert das
    # schon; bei direktem Aufruf mit clean_recordings-Daten kann aber NaN vorkommen.)
    sauber = [
        (s, l)
        for s, l in zip(sequences, labels)
        if np.isfinite(np.asarray(s, dtype=float)).all()
    ]
    sequences = [s for s, _ in sauber]
    labels = [l for _, l in sauber]

    # Jede Zustandszahl per Cross-Validation bewerten.
    results = [
        _cross_validate(sequences, labels, n, covariance_type, n_splits, random_state)
        for n in n_components_values
    ]

    # Nach Zustandszahl sortieren (fuer eine lesbare Tabelle).
    results.sort(key=lambda r: r["n_components"])

    # Beste Konfiguration: hoechste mittlere Genauigkeit; bei Gleichstand die
    # kleinere Streuung (stabiler ist besser).
    best = max(results, key=lambda r: (r["mean_accuracy"], -r["std_accuracy"]))
    return results, best
