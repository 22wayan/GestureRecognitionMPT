from SignalHub import GALY, get_nested_key, Module
from collections import deque
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
        - ``detector`` : Ergebnisse der Handdetektion

        Zusätzlich muss ein **Output-Schema** definiert werden.

        Output Schema
        -------------
        Das Modul erzeugt ein Signal mit dem Namen ``preprocessor``.

        Dieses Signal enthält entweder eine normalisierte Trajektorie
        oder ``None``, falls noch nicht genügend Daten gesammelt wurden.

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
        """
        super().__init__(
            inputSignals=["config", "detector"],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="preprocessor",
        )
        self.outputSignal = outputSignal

    def start(self, data):
        """
        Initialisierung des Modulzustands.

        Diese Methode wird einmal beim Start des Moduls ausgeführt.

        Ziel ist es, alle benötigten Parameter aus der Konfiguration zu
        lesen und interne Datenstrukturen vorzubereiten.

        Hinweise
        --------
        - Lese relevante Parameter aus der Konfiguration, z.B.
          den zu verfolgenden Finger.
        - Lege eine Datenstruktur an, um mehrere vergangene
          Fingerpositionen zu speichern, z.B. :class:`collections.deque`
          mit einer maximalen Größe.
        - Speichere außerdem Parameter wie die maximale Anzahl
          verlorener Frames oder die minimale Anzahl benötigter Punkte.
        - Zum Zugriff auf verschachtelte Konfigurationswerte kann
          :meth:`get_nested_key` verwendet werden.

        .. tip::
            Eine ``deque`` mit fester Länge ist ideal für Trajektorien,
            da alte Punkte automatisch verworfen werden.

        .. note::
            Trenne klar zwischen:
              - Initialisierung von Parametern (``start``)
              - Verarbeitung von Daten (``step``)

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
        # As a beginner student programmer, I need to initialize the module's state here.
        # First, I read the configuration values from the config signal.
        # I use get_nested_key to access nested config values safely.
        # finger_idx legt fest, welches Hand-Landmark verfolgt wird (Default 8 = Zeigefingerspitze).
        # WICHTIG: Der Schluessel heisst finger_idx -- genau wie in config.yml, im Training
        # (labeling.py) und im TrailMarker. Nur mit exakt gleichem Schluessel verfolgen Training
        # und Live denselben Finger; ein abweichender Name wuerde still auf den Default zurueckfallen.
        self.finger_idx = get_nested_key("config.preprocessor.finger_idx", data, 8)
        # buffer_size is the maximum number of points to store in the trajectory buffer.
        self.buffer_size = get_nested_key("config.preprocessor.buffer_size", data, 30)
        # min_steps is the minimum number of points needed for a valid trajectory.
        self.min_steps = get_nested_key("config.preprocessor.min_steps", data, 10)
        # max_lost is the maximum number of consecutive lost frames before ending the gesture.
        self.max_lost = get_nested_key("config.preprocessor.max_lost", data, 5)
        # min_speed_corner and reset_speed_corner are for hysteresis-based motion detection.
        # min_speed_corner is the threshold to start collecting when movement begins.
        self.min_speed_corner = get_nested_key("config.preprocessor.min_speed_corner", data, 0.01)
        # reset_speed_corner is the threshold to stop collecting when movement slows down.
        self.reset_speed_corner = get_nested_key("config.preprocessor.reset_speed_corner", data, 0.005)
        
        # Now, I create a deque to store the finger positions. It has a maximum length to keep only recent points.
        self.buffer = deque(maxlen=self.buffer_size)
        # lost_counter counts how many frames the hand has been lost.
        self.lost_counter = 0
        # is_moving indicates if the gesture is currently active based on hysteresis.
        self.is_moving = False
        
        # Return an empty dict as required.
        return {}

    def step(self, data):
        """
        Verarbeitung eines einzelnen Frames.

        Ziel ist es, eine Fingerposition aus den erkannten Landmarken
        zu extrahieren und diese in einer Trajektorie zu speichern.

        Hinweise
        --------
        - Greife auf das ``detector`` Signal zu, um erkannte
          Handlandmarks zu erhalten.
        - Falls keine Hand erkannt wurde, sollte ein interner
          Zähler für verlorene Frames erhöht werden.
        - Wird eine Hand erkannt, kann die Landmarke des gewünschten
          Fingers extrahiert werden.
        - Die Position dieses Fingers kann anschließend in einer
          Trajektorie gespeichert werden.
        - Sobald genügend Punkte gesammelt wurden, kann die
          Trajektorie weiterverarbeitet werden.

        Mögliche Verarbeitungsschritte:

        - Umwandlung der gespeicherten Punkte in ein
          :class:`numpy.ndarray`
        - Berechnung eines Zentrums der Trajektorie
        - Skalierung oder Normalisierung der Punkte

        .. tip::
            Arbeite schrittweise:
              1. Prüfen, ob Landmarken vorhanden sind
              2. Fingerposition extrahieren
              3. In Trajektorie speichern
              4. Optional normalisieren

        .. warning::
            Achte darauf, dass:
              - genügend Punkte vorhanden sind
              - keine fehlerhaften Frames verarbeitet werden
              - verlorene Frames sinnvoll behandelt werden

        Parameters
        ----------
        data : dict
            Enthält unter anderem:

            - ``detector`` : erkannte Hände und Landmarken
            - ``config`` : Systemkonfiguration

        Returns
        -------
        dict
            Gibt entweder ``None`` oder eine normalisierte Trajektorie
            zurück.

            Beispiel:

            ``return {outputSignal: trajectory}``
        """
        # As a beginner student, I implement the collection logic here.
        # First, get the detector data.
        detector = data.get('detector')
        if detector and detector.get('hands'):
            # If a hand is detected, extract the finger position.
            hand = detector['hands'][0]  # Assume the first hand.
            landmark = hand['landmarks'][self.finger_idx]
            pos = np.array([landmark['x'], landmark['y']])
            # Append the position to the buffer.
            self.buffer.append(pos)
            # Reset the lost counter since we have detection.
            self.lost_counter = 0
            # Check for movement using hysteresis.
            if len(self.buffer) > 1:
                # Calculate the speed as the norm of the difference.
                diff = np.linalg.norm(self.buffer[-1] - self.buffer[-2])
                if not self.is_moving and diff > self.min_speed_corner:
                    # Start collecting if movement detected.
                    self.is_moving = True
                elif self.is_moving and diff < self.reset_speed_corner:
                    # Stop collecting if movement slowed down.
                    self.is_moving = False
                    # Process the trajectory if enough points.
                    if len(self.buffer) >= self.min_steps:
                        trajectory = self.process_trajectory()
                        return {self.outputSignal: trajectory}
                    else:
                        # Discard if not enough points.
                        self.buffer.clear()
            # While collecting, emit None.
            return {self.outputSignal: None}
        else:
            # If no hand detected, increment lost counter.
            self.lost_counter += 1
            if self.lost_counter > self.max_lost and self.is_moving:
                # End the gesture if lost too many frames.
                self.is_moving = False
                if len(self.buffer) >= self.min_steps:
                    trajectory = self.process_trajectory()
                    return {self.outputSignal: trajectory}
                else:
                    self.buffer.clear()
            # Emit None.
            return {self.outputSignal: None}

    def process_trajectory(self):
        """
        Wandelt die gesammelten Rohpunkte in GENAU das Feature-Format um, mit dem
        der HMM-Klassifikator trainiert wurde: Spalten ``(x, y, velocity)``, also
        3 Zahlen (Features) pro Frame.

        Warum das so wichtig ist (fuer Anfaenger erklaert)
        --------------------------------------------------
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
        # -------------------------------------------------------------------
        # WICHTIG: Diese Umwandlung muss EXAKT das gleiche Ergebnis liefern wie
        # die Trainings-Pipeline in labeling.py. Sonst bekommt der Klassifikator
        # live ein anderes Datenformat als beim Training und kann die Geste nicht
        # bewerten (Score -inf -> live kommt immer "?" heraus).
        #
        # Damit Live und Training NIE wieder auseinanderlaufen, benutzen wir hier
        # GENAU DIESELBEN Funktionen, die auch beim Datensatz-Bau verwendet werden:
        #   _normalize   -> zentrieren + auf Einheitskreis skalieren  (x, y)
        #   _add_velocity -> Geschwindigkeit als 3. Spalte anhaengen   (x, y, velocity)
        #
        # Der Import steht bewusst HIER in der Methode (nicht oben in der Datei):
        # So kann kein zirkulaerer Import beim Programmstart entstehen. Die Methode
        # laeuft nur einmal pro fertiger Geste, der Import kostet also keine Zeit.
        # -------------------------------------------------------------------
        from GestureRecognition.labeling import _normalize, _add_velocity

        # 1) Aus den gesammelten Punkten (deque) ein numpy-Array (N, 2) machen.
        #    Jede Zeile ist eine (x, y)-Fingerposition in Bildkoordinaten [0, 1].
        trajectory = np.array(list(self.buffer), dtype=float)

        # 2) Normalisieren: Mittelpunkt abziehen und auf den Einheitskreis skalieren.
        #    Dadurch ist es egal, WO im Bild und WIE GROSS die Geste gemalt wurde.
        trajectory = _normalize(trajectory)

        # 3) Geschwindigkeit als dritte Spalte anhaengen -> (x, y, velocity).
        #    velocity = Abstand zum vorherigen Frame; der erste Frame bekommt 0.0.
        features = _add_velocity(trajectory)

        # 4) Puffer leeren, damit die naechste Geste sauber von vorne beginnt.
        self.buffer.clear()

        # Ergebnis: Array der Form (N, 3) mit Spalten (x, y, velocity) --
        # identisch zum Trainingsformat, das der HMMClassifier erwartet.
        return features

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
           wenn interne Zustände explizit zurückgesetzt werden sollen.

        Parameters
        ----------
        data : dict
            Letzte übergebene Daten des Frameworks.
        """
        pass