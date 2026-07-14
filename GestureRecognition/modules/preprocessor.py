from SignalHub import get_nested_key, Module
import numpy as np

class Preprocessor(Module):
    """
    Modul zur Vorverarbeitung von Fingertrajektorien.

    Dieses Modul verarbeitet die vom Handdetektor gelieferten Landmarken
    und extrahiert daraus die Bewegung eines bestimmten Fingers über
    mehrere Frames hinweg.

    Ziel ist es, eine Trajektorie zu sammeln, diese zu normalisieren
    und anschließend als Eingabe für nachfolgende Module bereitzustellen.
    """

    def __init__(self, outputSignal="preprocessor"):
        """
        Meldet das Modul beim SignalHub-Framework an.

        Wir abonnieren ``config`` (Einstellungen aus der config.yml) und
        ``detector`` (die erkannten Hände aus dem HandDetector). Als Ausgabe
        melden wir das Signal ``preprocessor`` an: Dort liegt entweder eine
        fertige, normalisierte Trajektorie oder ``None``, solange die Geste
        noch nicht abgeschlossen ist.

        Parameters
        ----------
        outputSignal : str, optional
            Name des erzeugten Output-Signals.
        """
        super().__init__(
            inputSignals=["config", "detector"],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="preprocessor",
        )
        self.outputSignal = outputSignal

    def start(self, data):
        """
        Initialisierung des Modulzustands (läuft einmal beim Start).

        Hier lesen wir alle Parameter für die Gesten-Segmentierung aus der
        Konfiguration und bauen daraus den gemeinsamen
        :class:`~GestureRecognition.labeling.GestureSegmenter`.

        Parameters
        ----------
        data : dict
            Eingabedaten des Frameworks. Enthält unter anderem das
            Signal ``config``.

        Returns
        -------
        dict
            Ein leeres Dictionary.
        """
        # finger_idx legt fest, welches Hand-Landmark verfolgt wird (Default 8 = Zeigefingerspitze).
        # WICHTIG: Der Schluessel heisst finger_idx -- genau wie in config.yml, im Training
        # (labeling.py) und im TrailMarker. Nur mit exakt gleichem Schluessel verfolgen Training
        # und Live denselben Finger; ein abweichender Name wuerde still auf den Default zurueckfallen.
        self.finger_idx = get_nested_key("config.preprocessor.finger_idx", data, 8)
        # Maximale Punktzahl im Puffer -- muss gross genug fuer langsame/grosse Buchstaben sein.
        self.buffer_size = get_nested_key("config.preprocessor.buffer_size", data, 30)
        # Weniger Punkte als min_steps gelten nicht als richtige Geste (nur Zittern o.ae.).
        self.min_steps = get_nested_key("config.preprocessor.min_steps", data, 10)
        # Nach so vielen Frames ohne Hand gilt die Geste als beendet.
        self.max_lost = get_nested_key("config.preprocessor.max_lost", data, 5)
        # Hysterese fuer die Bewegungs-Erkennung: Sammeln startet erst ueber
        # min_speed_corner und endet erst unter reset_speed_corner. Die Start-
        # Schwelle liegt bewusst UEBER der Stopp-Schwelle, damit die Erkennung
        # nicht bei jeder kleinen Tempo-Schwankung an- und ausgeht.
        self.min_speed_corner = get_nested_key("config.preprocessor.min_speed_corner", data, 0.01)
        self.reset_speed_corner = get_nested_key("config.preprocessor.reset_speed_corner", data, 0.005)
        # stop_hold: Wie viele langsame Frames HINTEREINANDER noetig sind, damit eine
        # Geste als beendet gilt (sogenanntes "Entprellen"). Ein einzelner langsamer
        # Frame -- etwa an der Ecke eines M oder Z, wo man kurz abbremst -- beendet die
        # Geste dann NICHT mehr; nur ein echtes Anhalten ueber mehrere Frames tut das.
        self.stop_hold = get_nested_key("config.preprocessor.stop_hold", data, 4)

        # Die eigentliche Segmentierungs-Logik (Hysterese, Entprellen, Handverlust)
        # lebt seit Issue #58 in GestureRecognition.labeling.GestureSegmenter --
        # als EINE gemeinsame Implementierung fuer Live-Betrieb und Datensatzbau.
        # Dieses Modul ist nur noch der duenne Live-Wrapper darum.
        from GestureRecognition.labeling import GestureSegmenter

        self.segmenter = GestureSegmenter(
            min_speed=self.min_speed_corner,
            reset_speed=self.reset_speed_corner,
            stop_hold=self.stop_hold,
            max_lost=self.max_lost,
            min_steps=self.min_steps,
            buffer_size=self.buffer_size,
        )

        return {}

    def step(self, data):
        """
        Verarbeitung eines einzelnen Frames.

        Wir bestimmen die aktuelle Fingerposition und geben sie an den
        gemeinsamen :class:`GestureSegmenter` weiter. Der entscheidet, wann
        eine Geste anfängt und aufhört. Sobald er ein fertiges Segment
        liefert, wandeln wir es in das Trainings-Feature-Format um und geben
        es als Signal weiter -- in allen anderen Frames geben wir ``None``.

        Parameters
        ----------
        data : dict
            Enthält unter anderem:

            - ``detector`` : erkannte Hände und Landmarken
            - ``config`` : Systemkonfiguration

        Returns
        -------
        dict
            ``{outputSignal: trajectory}`` mit der fertigen Trajektorie
            oder ``{outputSignal: None}``, solange keine Geste fertig ist.
        """
        # Fingerposition dieses Frames bestimmen (oder None = keine Hand) und an
        # die gemeinsame Segmentierungs-Logik weiterreichen. Start/Stopp/Entprellen/
        # Handverlust entscheidet der GestureSegmenter -- dieselbe Logik, die auch
        # der Datensatzbau benutzt (Issue #58).
        detector = data.get('detector')
        pos = None
        if detector and detector.get('hands'):
            # Wir nehmen immer die erste erkannte Hand.
            hand = detector['hands'][0]
            landmark = hand['landmarks'][self.finger_idx]
            pos = np.array([landmark['x'], landmark['y']])

        segment = self.segmenter.feed(pos)
        if segment is not None:
            # Eine Geste ist fertig -> in das Trainings-Feature-Format bringen.
            return {self.outputSignal: self.process_trajectory(segment)}
        # Keine fertige Geste in diesem Frame.
        return {self.outputSignal: None}

    def process_trajectory(self, segment):
        """
        Wandelt die gesammelten Rohpunkte in GENAU das Feature-Format um, mit dem
        der HMM-Klassifikator trainiert wurde: Spalten ``(x, y, velocity)``, also
        3 Zahlen (Features) pro Frame.

        Warum das so wichtig ist
        ------------------------
        Der Klassifikator lernt beim Training ein eigenes Modell pro Geste. Jedes
        Modell merkt sich dabei, WIE VIELE Features eine Trajektorie pro Zeitschritt
        hat und wie diese Zahlen ungefaehr verteilt sind. In der Live-Erkennung
        muessen wir dem Modell deshalb Daten im GENAU GLEICHEN Format geben -- sonst
        kann es die Sequenz nicht bewerten.

        Trainiert wird ueber :func:`GestureRecognition.labeling.clean_recordings`.
        Dort passiert pro Aufnahme: erst :func:`_normalize`, dann
        :func:`_add_velocity` -> Ergebnis ``(x, y, velocity)``.

        Der Fehler, der hier frueher steckte
        -------------------------------------
        Frueher erzeugte diese Methode ein ANDERES Format: 4 Features
        ``(x, y, dx, dy)`` und eine andere Normalisierung (Teilen durch die
        Standardabweichung statt durch den Einheitskreis). 4 Features passen aber
        nicht in ein Modell, das auf 3 Features trainiert wurde -> das Modell konnte
        die Live-Sequenz nicht bewerten (Score ``-inf``) -> live kam IMMER "?".

        Die Loesung: eine einzige Quelle der Wahrheit
        ---------------------------------------------
        Statt Normalisierung + Velocity hier ein zweites Mal (und leicht anders) zu
        programmieren, rufen wir dieselben Funktionen wie das Training auf. So
        koennen Training und Live nie wieder auseinanderlaufen.

        Returns
        -------
        np.ndarray
            Trajektorie der Form ``(N, 3)`` mit Spalten ``(x, y, velocity)`` --
            identisch zum Trainingsformat.
        """
        # Wir rufen exakt dieselbe Funktion auf wie der Datensatz-Bau beim
        # Training (_to_features: Resampling + Normalisierung + Geschwindigkeit).
        # So bekommt das Modell live garantiert dasselbe Format wie beim Lernen.
        # Der Import steht bewusst HIER in der Methode (nicht oben in der Datei),
        # damit beim Programmstart kein zirkulaerer Import entsteht -- die Methode
        # laeuft nur einmal pro fertiger Geste, das kostet also keine Zeit.
        from GestureRecognition.labeling import _to_features

        return _to_features(np.asarray(segment, dtype=float))

    def stop(self, data):
        """
        Wird beim Beenden aufgerufen. Wir halten keine externen Ressourcen,
        deshalb gibt es hier nichts aufzuräumen.
        """
        pass