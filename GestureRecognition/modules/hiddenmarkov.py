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
        Konstruktor des Moduls.

        Ziel ist es, das Modul beim Framework korrekt zu registrieren.

        Hinweise
        --------
        - Ein Modul muss definieren, **welche Signale es empfangen möchte**.
        - Diese werden über ``inputSignals`` angegeben.
        - Nur Signale, die hier subscribed werden, erscheinen später im
          ``data`` Dictionary der Methoden :meth:`start` und :meth:`step`.

        Für dieses Modul werden unter anderem folgende Signale benötigt:

        - ``config`` : Systemkonfiguration
        - ``preprocessor`` : normalisierte Trajektorien

        Zusätzlich muss ein **Output-Schema** definiert werden.

        Output Schema
        -------------
        Das Modul erzeugt ein Signal mit dem Namen ``markov``.

        Dieses Signal enthält Informationen über die erkannte Geste
        sowie deren Klassifikationsscore.

        Beispiel:

        ``outputSchema={"type": "object", "properties": {outputSignal: {}}}``

        .. note::
           Die Basisklasse :class:`Module` erwartet beim Aufruf von
           ``super().__init__`` unter anderem:

           - ``inputSignals``
           - ``outputSchema``
           - ``name`` des Moduls

        Parameters
        ----------
        outputSignal : str, optional
            Name des erzeugten Output-Signals.

        model_path : str, optional
            Pfad zu einem gespeicherten HMM-Modell.

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
        Initialisierung des Moduls.

        Diese Methode wird einmal beim Start des Moduls ausgeführt.

        Ziel ist es, ein zuvor trainiertes Hidden-Markov-Modell zu laden,
        das später zur Klassifikation verwendet wird.

        Hinweise
        --------
        - Das Modell kann aus einer Datei geladen werden.
        - Typischerweise wird dafür eine Klassenmethode verwendet,
          die ein gespeichertes Modell rekonstruiert.
        - Das geladene Modell sollte als Attribut des Moduls gespeichert
          werden, damit es in :meth:`step` verwendet werden kann.

        .. tip::
           Trenne klar zwischen:
            - Modell laden (``start``)
            - Modell anwenden (``step``)

        .. warning::
           Stelle sicher, dass:
            - der Pfad korrekt ist
            - das Modell zum erwarteten Datenformat passt

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

        Ziel ist es, eine vorverarbeitete Trajektorie zu klassifizieren
        und die wahrscheinlichste Geste zu bestimmen.

        Hinweise
        --------
        - Greife auf das ``preprocessor`` Signal zu.
        - Falls keine Trajektorie vorhanden ist, kann die Verarbeitung
          übersprungen werden.
        - Das geladene HMM-Modell kann anschließend verwendet werden,
          um eine Entscheidung für die aktuelle Bewegung zu berechnen.
        - Das Ergebnis enthält typischerweise Scores für mehrere Klassen.
        - Die Klasse mit dem höchsten Score kann als Ergebnis gewählt werden.

        Zusätzlich kann eine Visualisierung erzeugt werden:

        - Erzeuge ein :class:`GALY` Objekt.
        - Lege eine neue Zeichenebene an.
        - Verwende :meth:`putText`, um Score und Label darzustellen.
        - Für die Skalierung der Zeichenebene können Parameter aus der
          Konfiguration über :meth:`get_nested_key` gelesen werden.

        .. tip::
           Typischer Ablauf:
            1. Daten prüfen (existiert eine Sequenz?)
            2. Modell anwenden
            3. Scores interpretieren
            4. Ergebnis visualisieren

        .. note::
           Du entscheidest selbst:
            - wie du Scores darstellst
            - ob du nur das beste Label oder mehrere Kandidaten zeigst

        .. warning::
           Achte darauf, dass:
            - das Eingabeformat exakt zum Trainingsformat passt
            - keine leeren oder fehlerhaften Sequenzen verarbeitet werden

        Parameters
        ----------
        data : dict
            Enthält unter anderem:

            - ``preprocessor`` : normalisierte Trajektorie
            - ``config`` : Systemkonfiguration

        Returns
        -------
        dict
            Soll die erkannte Geste sowie optional Visualisierungsdaten
            enthalten.

            Beispiel:

            ``return {outputSignal: result, "galy": galy}``
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
        Wird aufgerufen, wenn das Modul beendet wird.

        Ziel ist es, bei Bedarf interne Zustände zurückzusetzen
        oder Ressourcen freizugeben.

        Hinweise
        --------
        - In vielen Fällen ist keine spezielle Bereinigung notwendig.

        .. note::
           Diese Methode ist optional, kann aber relevant werden,
           wenn Modelle oder externe Ressourcen verwaltet werden.

        Parameters
        ----------
        data : dict
            Letzte übergebene Daten des Frameworks.
        """
        pass
