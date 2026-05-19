from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from hmmlearn.hmm import GaussianHMM


class HMMClassifier:
    """
    TODO: Implementiere einen HMM-basierten Klassifikator

    Ziel:
    -----
    Entwickle einen Klassifikator, der zeitliche Sequenzen mit Hilfe von
    Hidden-Markov-Modellen (HMMs) klassifiziert. Für HMMs können libraries wie
    :mod:`hmmlearn` benutzt werden

    Grundidee:
    ----------
    - Trainiere ein Modell pro Klasse
    - Bewerte neue Sequenzen anhand der Likelihood unter jedem Modell
    - Wähle die Klasse mit der höchsten Wahrscheinlichkeit

    .. note::
       Wie genau deine Modelle aussehen (z. B. Anzahl Zustände, Features,
       Initialisierung etc.) ist bewusst nicht vorgegeben.

    Wichtige Designentscheidungen:
    ------------------------------
    - Wie strukturierst du deine Trainingsdaten?
    - Wie repräsentierst du Sequenzen?
    - Wie verbindest du mehrere Sequenzen mit Labels?

    Speicherung:
    ------------
    Du solltest dir überlegen:
    - Wie speicherst du dein trainiertes Modell?
    - Wie lädst du es später wieder?
    - Welche Informationen müssen persistiert werden (z. B. Klassen, Modelle)?

    .. tip::
       ``pickle`` ist eine einfache Möglichkeit, Modelle zu speichern.
       Alternativ kannst du auch eigene Formate definieren.

    Evaluation:
    -----------
    Für sinnvolles Training solltest du unbedingt:
    - eine eigene ``train_test_split``-Logik implementieren
    - Trainings- und Testdaten sauber trennen

    .. warning::
       Wenn du Training und Test nicht trennst, sind deine Ergebnisse nicht aussagekräftig.

    Erweiterung (optional):
    -----------------------
    - Implementiere eine Grid Search für Hyperparameter
      (z. B. Anzahl Zustände, Modellstruktur)
    - Vergleiche verschiedene Modellkonfigurationen

    """

    def __init__(
        self,
        n_components=4,
        covariance_type="diag",
        random_state=42,
        n_iter=100,
        min_covar=1e-3,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.n_iter = n_iter
        self.min_covar = min_covar
        self.models_ = {}
        self.classes_ = []

    def _as_sequence(self, sequence):
        sequence = np.asarray(sequence, dtype=float)
        if sequence.ndim == 1:
            sequence = sequence.reshape(-1, 1)
        if sequence.ndim != 2:
            raise ValueError("Jede Sequenz muss 2-dimensional sein.")
        if len(sequence) == 0:
            raise ValueError("Leere Sequenzen sind nicht erlaubt.")
        return sequence

    def _prepare_training_data(self, X, y=None):
        if isinstance(X, dict):
            if "sequences" in X and "labels" in X:
                sequences = X["sequences"]
                labels = X["labels"]
            elif "X" in X and "y" in X:
                sequences = X["X"]
                labels = X["y"]
            else:
                sequences = []
                labels = []
                for label, seqs in X.items():
                    for seq in seqs:
                        sequences.append(seq)
                        labels.append(label)
        else:
            sequences = X
            labels = y

        if labels is None:
            raise ValueError("Für das Training werden Labels benötigt.")

        if len(sequences) != len(labels):
            raise ValueError("Anzahl von Sequenzen und Labels passt nicht zusammen.")

        prepared_sequences = [self._as_sequence(sequence) for sequence in sequences]
        prepared_labels = [str(label) for label in labels]
        return prepared_sequences, prepared_labels

    def _prepare_inference_data(self, X):
        if isinstance(X, np.ndarray):
            if X.ndim == 2:
                return [self._as_sequence(X)]
            if X.ndim == 3:
                return [self._as_sequence(sequence) for sequence in X]
            raise ValueError("NumPy-Eingaben muessen 2D oder 3D sein.")

        if isinstance(X, (list, tuple)):
            if not X:
                return []

            first_item = np.asarray(X[0])
            if first_item.ndim <= 1:
                return [self._as_sequence(X)]

            return [self._as_sequence(sequence) for sequence in X]

        return [self._as_sequence(X)]

    def fit(self, X, y=None):
        """
        TODO: Trainiere den Klassifikator

        Ziel:
        -----
        Trainiere ein separates HMM für jede Klasse basierend auf den
        gegebenen Sequenzen.


        Anforderungen / Ideen:
        ----------------------
        - Zerlege die Daten so, dass du pro Klasse alle Sequenzen bekommst
        - Trainiere ein Modell pro Klasse
        - Speichere die trainierten Modelle intern

        .. tip::
           Überlege dir eine sinnvolle Datenstruktur wie:
           ``label -> (Daten, Sequenzlängen)``

        .. note::
           Die konkrete Umsetzung ist offen:
            - Wie genau du Daten aufteilst
            - Wie du dein Modell initialisierst
            - Welche Hyperparameter du verwendest

        .. warning::
           Achte darauf, dass:
            - ``lengths`` zu ``X`` passen
            - Labels korrekt zu Sequenzen zugeordnet sind

        Erweiterung:
        ------------
        - Experimentiere mit verschiedenen Modellgrößen
        - Nutze eine Grid Search zur Optimierung
        - Verwende ein separates Testset zur Evaluation

        Returns
        -------
        self
        """
        sequences, labels = self._prepare_training_data(X, y)
        grouped_sequences = defaultdict(list)

        for sequence, label in zip(sequences, labels):
            grouped_sequences[label].append(sequence)

        self.models_ = {}
        self.classes_ = sorted(grouped_sequences.keys())

        for label in self.classes_:
            class_sequences = grouped_sequences[label]
            lengths = [len(sequence) for sequence in class_sequences]
            X_label = np.vstack(class_sequences)

            # Sehr einfache Schutzmaßnahme, damit kurze Sequenzen das Training
            # nicht sofort sprengen.
            n_components = max(1, min(self.n_components, min(lengths)))

            model = GaussianHMM(
                n_components=n_components,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                n_iter=self.n_iter,
                min_covar=self.min_covar,
            )
            model.fit(X_label, lengths)
            self.models_[label] = model

        return self

    def decision_function(self, X):
        """
        TODO: Berechne Scores für jede Klasse

        Ziel:
        -----
        Berechne für jede Eingabesequenz einen Score pro Klasse
        (z. B. Log-Likelihood unter jedem Modell).

        Anforderungen / Ideen:
        ----------------------
        - Zerlege die Eingabe in einzelne Sequenzen
        - Berechne für jede Sequenz:
            Score unter jedem Klassenmodell
        - Gib eine Struktur zurück wie:
            ``(n_sequences, n_classes)``

        .. tip::
           Die meisten HMM-Implementierungen bieten eine
           ``score``-Funktion für Likelihoods.

        .. note::
           Du entscheidest selbst:
            - Welcher Score verwendet wird
            - Wie du mehrere Sequenzen behandelst

        .. warning::
           Stelle sicher, dass:
            - Die Reihenfolge der Klassen konsistent ist
            - Scores vergleichbar sind

        Returns
        -------
        scores : array-like
            Score pro Sequenz und Klasse
        """
        if not self.models_:
            raise ValueError("Der Klassifikator wurde noch nicht trainiert.")

        sequences = self._prepare_inference_data(X)
        if not sequences:
            return np.empty((0, len(self.classes_)))

        scores = np.full((len(sequences), len(self.classes_)), -np.inf, dtype=float)

        for sequence_idx, sequence in enumerate(sequences):
            for class_idx, label in enumerate(self.classes_):
                try:
                    scores[sequence_idx, class_idx] = self.models_[label].score(sequence)
                except Exception:
                    scores[sequence_idx, class_idx] = -np.inf

        return scores

    def predict(self, X):
        """
        TODO: Sage Klassenlabels voraus

        Ziel:
        -----
        Weise jeder Eingabesequenz ein Label zu.

        Anforderungen / Ideen:
        ----------------------
        - Nutze deine ``decision_function``
        - Wähle für jede Sequenz die Klasse mit bestem Score

        .. tip::
           Typischerweise:
           ``argmax über Klassen``

        .. note::
           Achte darauf, dass:
            - Klassenreihenfolge konsistent ist
            - Rückgabewerte klar interpretierbar sind

        Erweiterung:
        ------------
        - Gib zusätzlich Unsicherheiten oder Scores zurück
        - Implementiere Top-k Vorhersagen

        Returns
        -------
        labels : list
            Vorhergesagte Labels
        """
        scores = self.decision_function(X)
        if scores.size == 0:
            return []

        best_indices = np.argmax(scores, axis=1)
        return [self.classes_[index] for index in best_indices]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        with Path(path).open("rb") as file:
            loaded_object = pickle.load(file)

        if isinstance(loaded_object, cls):
            return loaded_object

        raise TypeError("Die geladene Datei enthaelt keinen HMMClassifier.")
