from pathlib import Path

import numpy as np
from SignalHub import GALY, Module

from GestureRecognition.hmmclassifier import HMMClassifier


class HMMModule(Module):
    """
    Modul zur Klassifikation von Gesten mittels Hidden Markov Models.

    Dieses Modul erhält eine vorverarbeitete Fingertrajektorie vom
    :class:`Preprocessor` Modul und verwendet ein trainiertes
    Hidden-Markov-Modell, um eine Geste zu klassifizieren.

    Ziel ist es, eine geladene Modellstruktur zu verwenden, um
    eine Entscheidung über die aktuell ausgeführte Bewegung zu treffen
    und das Ergebnis an das Framework zurückzugeben.
    """

    def __init__(self, outputSignal="markov", model_path="data/hmm.pkl", **kwargs):
        """
        Meldet das Modul beim SignalHub-Framework an.

        Wir abonnieren ``config`` (Einstellungen) und ``preprocessor``
        (die fertige, normalisierte Trajektorie einer Geste). Als Ausgabe
        melden wir das Signal ``markov`` an, in dem das Klassifikations-
        ergebnis (Label, Score, Margin) landet, plus ``galy`` für die
        Anzeige im Kamerabild.

        Parameters
        ----------
        outputSignal : str, optional
            Name des erzeugten Output-Signals.

        model_path : str, optional
            Pfad zu einem gespeicherten HMM-Modell (Standard: data/hmm.pkl,
            wird von train.py erzeugt).

        **kwargs
            Weitere Parameter, die an :class:`Module` weitergegeben werden.
        """
        super().__init__(
            inputSignals=["config", "preprocessor"],
            outputSchema={
                "type": "object",
                "properties": {
                    outputSignal: {},
                    "galy": {},
                },
            },
            name="hiddenmarkov",
        )
        self.outputSignal = outputSignal
        self.model_path = Path(model_path)
        self.classifier = None
        self.score_threshold = -20.0
        self.margin_threshold = 0.5
        self.unknown_label = "?"
        # Letztes Klassifikationsergebnis merken, damit das Anzeigefenster den
        # zuletzt erkannten Buchstaben DAUERHAFT zeigt. Ohne das wuerde nur in dem
        # einen Frame, in dem eine Geste fertig wird, kurz Text erscheinen -- das
        # Fenster bliebe sonst die meiste Zeit schwarz.
        self.last_label = "..."
        self.last_score = 0.0
        self.last_margin = 0.0

    def start(self, data):
        """
        Initialisierung des Moduls (läuft einmal beim Start).

        Hier lesen wir die Schwellwerte aus der Konfiguration und laden das
        trainierte HMM-Modell von der Festplatte. Fehlt die Modelldatei
        (z.B. weil ``python train.py`` noch nie lief), bleibt der
        Klassifikator ``None`` und das Modul zeigt dauerhaft nur "...".

        Parameters
        ----------
        data : dict
            Eingabedaten des Frameworks.

        Returns
        -------
        dict
            Ein leeres Dictionary.
        """
        config = data.get("config", {})
        hiddenmarkov_config = config.get("hiddenmarkov", {})

        self.score_threshold = hiddenmarkov_config.get("score_threshold", self.score_threshold)
        self.margin_threshold = hiddenmarkov_config.get(
            "margin_threshold", self.margin_threshold
        )
        self.unknown_label = hiddenmarkov_config.get("unknown_label", self.unknown_label)

        if self.model_path.exists():
            self.classifier = HMMClassifier.load(self.model_path)
        else:
            self.classifier = None

        return {}

    def _build_galy(self, config, label, score, margin):
        """Baut die Text-Anzeige (Label, Score, Margin) für das Kamerabild."""
        galy = GALY()

        # Das Label wird als Overlay-EBENE direkt auf das Kamerabild "Main"
        # gezeichnet -- genau wie HandDetector ("hands") und TrailMarker ("trail")
        # es tun. Frueher legte dieses Modul eine EIGENE Flaeche "main" an; die
        # wurde als separates Fenster geoeffnet, das leer/schwarz blieb. Als Ebene
        # auf "Main" erscheint das Ergebnis dort, wo der Nutzer ohnehin hinschaut.
        galy.layer("hiddenmarkov")

        text_lines = [
            f"Label: {label}",
            f"Score: {score:.3f}",
            f"Margin: {margin:.3f}",
        ]

        for index, text in enumerate(text_lines):
            galy.putText(text, (10, 30 + index * 30), color=(255, 255, 255))

        return galy

    def _show_last(self, config):
        """Anzeige der ZULETZT erkannten Geste (ohne neues Ergebnis).

        Wird in jedem Frame benutzt, in dem gerade keine Geste fertig wird, damit
        das Anzeigefenster den letzten Buchstaben weiter zeigt statt schwarz zu
        werden.
        """
        return {
            self.outputSignal: None,
            "galy": self._build_galy(
                config, self.last_label, self.last_score, self.last_margin
            ),
        }

    def step(self, data):
        """
        Verarbeitung eines einzelnen Frames.

        In den meisten Frames liefert der Preprocessor ``None`` -- dann
        zeigen wir einfach das letzte Ergebnis weiter an. Kommt eine fertige
        Trajektorie an, lassen wir das HMM-Modell alle Klassen bewerten,
        nehmen die Klasse mit dem besten Score und prüfen mit zwei
        Schwellwerten, ob wir dem Ergebnis trauen: Ist der Score zu schlecht
        oder der Abstand zum zweitbesten Kandidaten (Margin) zu klein,
        geben wir lieber "?" (unbekannt) aus.

        Parameters
        ----------
        data : dict
            Enthält unter anderem:

            - ``preprocessor`` : normalisierte Trajektorie
            - ``config`` : Systemkonfiguration

        Returns
        -------
        dict
            ``{outputSignal: result, "galy": galy}`` mit Label, Score,
            Margin und allen Klassen-Scores.
        """
        trajectory = data.get("preprocessor")
        config = data.get("config", {})

        if self.classifier is None or trajectory is None:
            # Keine neue Geste in diesem Frame -> die letzte Erkennung weiter
            # anzeigen, damit das Fenster nicht schwarz wird.
            return self._show_last(config)

        trajectory = np.asarray(trajectory, dtype=float)
        if trajectory.ndim == 1:
            trajectory = trajectory.reshape(-1, 1)

        if trajectory.ndim != 2 or len(trajectory) == 0:
            # Keine neue Geste in diesem Frame -> die letzte Erkennung weiter
            # anzeigen, damit das Fenster nicht schwarz wird.
            return self._show_last(config)

        lengths = [len(trajectory)]
        scores = self.classifier.decision_function(trajectory, lengths)
        if scores.size == 0:
            # Keine neue Geste in diesem Frame -> die letzte Erkennung weiter
            # anzeigen, damit das Fenster nicht schwarz wird.
            return self._show_last(config)

        # Die Log-Scores haengen von der Laenge der Sequenz ab (mehr Frames =
        # kleinerer Score). Wir teilen deshalb durch die Laenge, damit kurze und
        # lange Gesten mit denselben Schwellwerten vergleichbar sind.
        score_vector = scores[0]
        normalized_scores = score_vector / max(len(trajectory), 1)
        best_index = int(np.argmax(normalized_scores))
        sorted_scores = np.sort(normalized_scores)[::-1]
        best_score = float(normalized_scores[best_index])
        second_best_score = float(sorted_scores[1]) if len(sorted_scores) > 1 else float("-inf")
        margin = float(best_score - second_best_score) if np.isfinite(second_best_score) else float("inf")

        best_label = self.classifier.classes_[best_index]
        is_unknown = best_score < self.score_threshold or margin < self.margin_threshold
        label = self.unknown_label if is_unknown else best_label

        all_scores = {
            class_label: float(class_score)
            for class_label, class_score in zip(self.classifier.classes_, normalized_scores)
        }

        result = {
            "label": label,
            "best_label": best_label,
            "score": best_score,
            "margin": margin,
            "scores": all_scores,
            "unknown": is_unknown,
            "thresholds": {
                "score_threshold": self.score_threshold,
                "margin_threshold": self.margin_threshold,
            },
        }

        # Neue Erkennung merken, damit sie in den folgenden Frames weiter
        # angezeigt wird (bis die naechste Geste kommt).
        self.last_label = label
        self.last_score = best_score
        self.last_margin = margin

        return {
            self.outputSignal: result,
            "galy": self._build_galy(config, label, best_score, margin),
        }

    def stop(self, data):
        """
        Wird beim Beenden aufgerufen. Das Modell liegt nur im Speicher,
        deshalb gibt es hier nichts aufzuräumen.
        """
        pass
