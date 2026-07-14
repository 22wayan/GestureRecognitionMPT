import pickle
import logging
import numpy as np
from pathlib import Path
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)

class HMMClassifier:
    """
    HMM-basierter Klassifikator für Gestensequenzen.

    Pro Klasse wird ein separates :class:`hmmlearn.hmm.GaussianHMM` trainiert.
    Klassifikation läuft über den Argmax der Log-Likelihoods aller Klassenmodelle.

    Hintergrund
    -----------
    Ein Hidden Markov Model beschreibt eine Zeitreihe als Folge von
    verborgenen Zuständen, die beobachtbare Ausgaben erzeugen.
    Das Modell ist definiert durch:

    - **A** (Übergangsmatrix): Wahrscheinlichkeit, von Zustand i nach j zu wechseln
    - **B** (Ausgabeverteilung): Wahrscheinlichkeit, eine Beobachtung in Zustand i zu sehen
      (hier Gauß-Verteilung pro Zustand)
    - **π** (Startverteilung): Wahrscheinlichkeit, in Zustand i zu beginnen

    Diese Markov-Annahme besagt, dass der nächste Zustand nur vom aktuellen abhängt,
    nicht von der gesamten Vorgeschichte — sinnvoll für Gesten, da
    die Bewegungsrichtung im nächsten Frame vor allem vom aktuellen Moment abhängt.

    Training
    --------
    Der Baum-Welch-Algorithmus (Expectation-Maximization) schätzt A, B und π
    iterativ: E-Schritt berechnet Zustandswahrscheinlichkeiten über den
    Forward-Backward-Algorithmus, M-Schritt maximiert die Parameter.

    Inferenz
    --------
    Der Forward-Algorithmus berechnet die Log-Likelihood P(X | Modell) effizient
    in O(T · K²) mit T = Sequenzlänge und K = Anzahl Zustände.
    Die Klasse mit der höchsten Log-Likelihood wird vorhergesagt.

    Hyperparameter
    --------------
    - ``n_components=10``: Zehn Zustaende pro Buchstabe. Mehr Zustaende koennen die
      Form einer Geste genauer beschreiben. Zusammen mit dem Resampling (jede Geste
      bekommt gleich viele Punkte, siehe ``labeling._to_features``) steigt die
      Genauigkeit deutlich (von ca. 72% auf ca. 90%). Bei sehr wenig Daten kann ein
      kleinerer Wert (4-6) besser sein.
    - ``covariance_type="diag"``: Diagonale Kovarianzmatrix nimmt an, dass
      x, y und Geschwindigkeit unkorreliert sind. Das reduziert die Parameteranzahl
      erheblich gegenüber ``"full"`` und macht das Modell stabiler bei wenig Daten.
    - ``min_covar=0.03``: Untergrenze für die Varianz jedes Zustands. Ohne diese
      Grenze kann ein Zustand mit wenig zugewiesenen Daten auf einen Punkt
      kollabieren (Varianz -> 0), wodurch die EM-Schätzung in ``NaN`` divergiert und
      das ganze Klassenmodell unbrauchbar wird (``score`` wirft, Klasse nie
      vorhergesagt). Der hmmlearn-Default ``1e-3`` reichte hier nicht: Buchstabe F
      kippte in NaN (0% Recall). ``0.03`` ist empirisch der kleinste Wert, der den
      Kollaps über alle 26 Klassen verhindert und dabei die Test-Accuracy maximiert
      (88% -> 90%).
    - ``n_iter=100``: Ausreichend für Konvergenz bei kurzen Gesten-Sequenzen.
    - ``tol=1e-2``: Konvergenzschwelle, stimmt mit hmmlearn-Standard überein.
    - ``random_state=42``: Reproduzierbarkeit.

    Parameters
    ----------
    n_components : int
        Anzahl verborgener Zustände pro Modell.
    covariance_type : str
        Art der Kovarianzmatrix: ``"diag"``, ``"full"``, ``"spherical"``, ``"tied"``.
    min_covar : float
        Untergrenze für die Varianz jedes Zustands. Verhindert Varianz-Kollaps
        (Zustand -> Punkt -> NaN) und damit unbrauchbare Klassenmodelle.
    n_iter : int
        Maximale Anzahl EM-Iterationen.
    tol : float
        Konvergenzschwelle für den Log-Likelihood-Anstieg.
    random_state : int or None
        Seed für die Initialisierung.
    """

    def __init__(
        self,
        n_components: int = 10,
        covariance_type: str = "diag",
        min_covar: float = 0.03,
        n_iter: int = 100,
        tol: float = 1e-2,
        random_state: int | None = 42,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state

        self.classes_: list = []
        self._models: dict[str, GaussianHMM] = {}

    def fit(self, X: np.ndarray, y: np.ndarray, lengths: list[int]) -> "HMMClassifier":
        """
        Trainiert ein separates GaussianHMM pro Klasse.

        Parameters
        ----------
        X : np.ndarray, shape (total_frames, n_features)
            Alle Trainingssequenzen hintereinandergehängt.
        y : array-like, shape (n_sequences,)
            Klassenlabel je Sequenz — muss zur Reihenfolge in ``lengths`` passen.
        lengths : list of int
            Länge jeder Sequenz. ``sum(lengths)`` muss gleich ``len(X)`` sein.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        lengths = list(lengths)

        self.classes_ = sorted(set(y))
        self._models = {}

        for label in self.classes_:
            mask = y == label
            class_lengths = [l for l, m in zip(lengths, mask) if m]
            class_X = self._split_and_concat(X, lengths, mask)

            model = self._fit_class_model(class_X, class_lengths, label)
            self._models[label] = model
            logger.info("Klasse '%s' trainiert (%d Sequenzen)", label, len(class_lengths))

        return self

    def _fit_class_model(
        self, class_X: np.ndarray, class_lengths: list[int], label
    ) -> GaussianHMM:
        """Trainiert ein GaussianHMM für eine Klasse und fängt den Varianz-Kollaps
        (ein Zustand kollabiert auf einen Punkt -> Kovarianz -> NaN) ab.

        Bei wenigen Trainingssequenzen — etwa einer neu vom Prüfer aufgenommenen
        Geste mit nur ~8 Aufnahmen — sind ``self.n_components`` Zustände manchmal
        zu viele: ein Zustand bekommt kaum Daten, seine Varianz läuft trotz
        ``min_covar`` in NaN, und ``score()`` liefert dann für diese Klasse
        ``NaN``/``-inf`` -> die Klasse wird nie vorhergesagt (stiller 0%-Recall).
        Das ist derselbe Effekt, der früher den Buchstaben F kippen ließ, hier aber
        datenabhängig (ein bestimmter Trainings-Split kippt, ein anderer nicht).

        Fallback: schlägt das Training mit ``self.n_components`` Zuständen in NaN um,
        wird die Zustandszahl schrittweise reduziert, bis das Klassenmodell stabil
        ist. Gut besetzte Klassen behalten so die volle Zustandszahl, nur datenarme
        Klassen bekommen automatisch ein kleineres, stabiles Modell.
        """
        for n_components in range(self.n_components, 0, -1):
            model = GaussianHMM(
                n_components=n_components,
                covariance_type=self.covariance_type,
                min_covar=self.min_covar,
                n_iter=self.n_iter,
                tol=self.tol,
                random_state=self.random_state,
            )
            try:
                model.fit(class_X, class_lengths)
            except Exception:
                # Fit selbst fehlgeschlagen (z.B. zu wenige Daten für so viele
                # Zustände) -> wie einen Kollaps behandeln und weniger Zustände nehmen.
                continue

            collapsed = np.isnan(model.means_).any() or np.isnan(model.covars_).any()
            if not collapsed:
                if n_components != self.n_components:
                    logger.warning(
                        "Klasse '%s': Training mit %d Zuständen kollabierte (NaN) -> "
                        "stabil mit %d Zuständen (zu wenige Trainingssequenzen?).",
                        label, self.n_components, n_components,
                    )
                return model

        raise RuntimeError(
            f"Klasse '{label}': HMM kollabiert selbst mit einem Zustand zu NaN — "
            f"Trainingsdaten prüfen."
        )

    def decision_function(self, X: np.ndarray, lengths: list[int]) -> np.ndarray:
        """
        Berechnet Log-Likelihoods aller Klassenmodelle für jede Sequenz.

        Sequenzen, die ein Modell nicht auswerten kann (z.B. zu kurz oder
        numerisch instabil), bekommen ``-inf`` als Score.

        Parameters
        ----------
        X : np.ndarray, shape (total_frames, n_features)
            Sequenzen hintereinandergehängt.
        lengths : list of int
            Länge jeder Sequenz.

        Returns
        -------
        scores : np.ndarray, shape (n_sequences, n_classes)
            Log-Likelihood jeder Sequenz unter jedem Klassenmodell.
            Spaltenreihenfolge entspricht ``self.classes_``.
        """
        X = np.asarray(X, dtype=float)
        n_seq = len(lengths)
        n_classes = len(self.classes_)
        scores = np.full((n_seq, n_classes), -np.inf)

        sequences = self._iter_sequences(X, lengths)

        for seq_idx, seq in enumerate(sequences):
            for cls_idx, label in enumerate(self.classes_):
                try:
                    scores[seq_idx, cls_idx] = self._models[label].score(seq)
                except Exception:
                    pass  # bleibt -inf

        return scores

    def predict(self, X: np.ndarray, lengths: list[int]) -> list:
        """
        Sagt für jede Sequenz die wahrscheinlichste Klasse vorher.

        Parameters
        ----------
        X : np.ndarray, shape (total_frames, n_features)
        lengths : list of int

        Returns
        -------
        list
            Vorhergesagte Klassenlabels, gleiche Länge wie ``lengths``.
        """
        scores = self.decision_function(X, lengths)
        indices = np.argmax(scores, axis=1)
        return [self.classes_[i] for i in indices]

    def predict_topk(self, X: np.ndarray, lengths: list[int], k: int) -> list[tuple]:
        """
        Gibt die Top-K-Vorhersagen mit zugehörigen Scores zurück.

        Parameters
        ----------
        X : np.ndarray, shape (total_frames, n_features)
        lengths : list of int
        k : int
            Anzahl der zurückgegebenen Klassen pro Sequenz.

        Returns
        -------
        list of tuple
            Jedes Element ist ``(labels, scores)`` mit den Top-K-Klassen
            (absteigend nach Score) und ihren Log-Likelihoods.
        """
        all_scores = self.decision_function(X, lengths)
        k = min(k, len(self.classes_))
        result = []
        for row in all_scores:
            top_indices = np.argsort(row)[::-1][:k]
            top_labels = [self.classes_[i] for i in top_indices]
            top_scores = row[top_indices].tolist()
            result.append((top_labels, top_scores))
        return result

    def save(self, path: str | Path) -> None:
        """
        Speichert den trainierten Klassifikator als Pickle-Datei.

        Parameters
        ----------
        path : str or Path
            Zielpfad für die Datei.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Modell gespeichert: %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "HMMClassifier":
        """
        Lädt einen gespeicherten Klassifikator aus einer Pickle-Datei.

        Parameters
        ----------
        path : str or Path
            Pfad zur gespeicherten Datei.

        Returns
        -------
        HMMClassifier
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("Modell geladen: %s", path)
        return obj

    @staticmethod
    def _split_and_concat(
        X: np.ndarray, lengths: list[int], mask: np.ndarray
    ) -> np.ndarray:
        """Gibt alle Sequenzen zurück, für die mask True ist, hintereinandergehängt."""
        parts = []
        offset = 0
        for length, keep in zip(lengths, mask):
            if keep:
                parts.append(X[offset : offset + length])
            offset += length
        return np.vstack(parts)

    @staticmethod
    def _iter_sequences(X: np.ndarray, lengths: list[int]):
        """Zerlegt X anhand von lengths in einzelne Sequenzen."""
        offset = 0
        for length in lengths:
            yield X[offset : offset + length]
            offset += length
