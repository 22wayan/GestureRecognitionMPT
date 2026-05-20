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
        # The finger_index specifies which finger landmark to track, default is 8 for index finger tip.
        self.finger_index = get_nested_key(data['config'], ['preprocessor', 'finger_index'], 8)
        # buffer_size is the maximum number of points to store in the trajectory buffer.
        self.buffer_size = get_nested_key(data['config'], ['preprocessor', 'buffer_size'], 30)
        # min_steps is the minimum number of points needed for a valid trajectory.
        self.min_steps = get_nested_key(data['config'], ['preprocessor', 'min_steps'], 10)
        # max_lost is the maximum number of consecutive lost frames before ending the gesture.
        self.max_lost = get_nested_key(data['config'], ['preprocessor', 'max_lost'], 5)
        # min_speed_corner and reset_speed_corner are for hysteresis-based motion detection.
        # min_speed_corner is the threshold to start collecting when movement begins.
        self.min_speed_corner = get_nested_key(data['config'], ['preprocessor', 'min_speed_corner'], 0.01)
        # reset_speed_corner is the threshold to stop collecting when movement slows down.
        self.reset_speed_corner = get_nested_key(data['config'], ['preprocessor', 'reset_speed_corner'], 0.005)
        
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
            landmark = hand['landmarks'][self.finger_index]
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
        Process the collected trajectory: normalize and add velocity features.
        
        Normalization strategy: Centering by subtracting the mean to center the trajectory around the origin,
        making it translation-invariant. Scaling by dividing by standard deviation + epsilon (1e-8) to make it 
        scale-invariant and stable against static trajectories where std=0.
        
        Used features: (x, y, dx, dy) representation, where x,y are normalized positions and dx,dy are velocities 
        computed using np.diff. This captures both spatial position and temporal motion for better gesture recognition.
        
        Threshold choices: min_steps=10 ensures a minimum trajectory length for meaningful analysis; max_lost=5 allows 
        brief hand losses without discarding the gesture; min_speed_corner=0.01 and reset_speed_corner=0.005 provide 
        hysteresis to robustly detect gesture start and end based on movement speed.
        """
        # Convert buffer to numpy array.
        trajectory = np.array(list(self.buffer))
        # Centering: subtract the mean.
        mean = np.mean(trajectory, axis=0)
        trajectory -= mean
        # Scaling: divide by std + epsilon to avoid division by zero.
        std = np.std(trajectory, axis=0) + 1e-8
        trajectory /= std
        # Velocity features: use np.diff to get differences.
        if len(trajectory) > 1:
            diffs = np.diff(trajectory, axis=0)
            # Combine positions and velocities: (x, y, dx, dy)
            features = np.column_stack((trajectory[:-1], diffs))
        else:
            # If only one point, just return it (though min_steps should prevent this).
            features = trajectory
        # Clear the buffer for next gesture.
        self.buffer.clear()
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