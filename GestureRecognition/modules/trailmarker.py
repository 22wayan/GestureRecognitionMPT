from SignalHub import Module, get_nested_key, GALY
from collections import deque
import numpy as np

class TrailMarker(Module):
    """
    Modul zum Zeichnen einer Spur anhand der Bewegung eines Fingers.

    Die Position eines bestimmten Finger-Landmarks wird über mehrere Frames
    hinweg gespeichert. Aus diesen Punkten kann anschließend eine Linie
    erzeugt werden, die den Bewegungsverlauf des Fingers visualisiert.

    Ziel ist es, die Verarbeitung der Landmark-Daten sowie die Verwaltung
    eines Zustands über mehrere Frames hinweg selbst zu implementieren.
    """

    def __init__(self, outputSignal="trailmarker"):
        """
        Meldet das Modul beim SignalHub-Framework an.

        Wir abonnieren ``config`` (Einstellungen aus der config.yml) und
        ``detector`` (die erkannten Hände aus dem HandDetector). Eigene Daten
        gibt das Modul nicht weiter -- es zeichnet nur die Spur ins Bild,
        deshalb bleibt das Output-Schema praktisch leer.

        Parameters
        ----------
        outputSignal : str, optional
            Name des erzeugten Output-Signals.
        """
        super().__init__(
            inputSignals=["config", "detector"],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="trailmarker",
        )

    def start(self, data):
        """
        Initialisierung des Modulzustands (läuft einmal beim Start).

        Hier lesen wir unsere Parameter aus der Konfiguration (welcher
        Finger verfolgt wird, wie lang die Spur sein darf) und legen die
        Datenstrukturen an, die wir über die Frames hinweg brauchen.

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
        # Unsere Einstellungen aus der config.yml holen.
        self.finger_idx = get_nested_key("config.trailmarker.finger_idx", data)
        self.max_lost = get_nested_key("config.trailmarker.max_lost", data)
        self.max_trail_length = get_nested_key("config.trailmarker.max_trail_length", data)
        self.webcam_width = get_nested_key("config.webcam.width", data)
        self.webcam_height = get_nested_key("config.webcam.height", data)

        # Eine deque mit fester Maximallaenge wirft alte Punkte automatisch
        # raus -- so waechst die Spur nie unendlich.
        self.trail = deque(maxlen=self.max_trail_length)

        # Zaehlt, in wie vielen Frames hintereinander keine Hand zu sehen war.
        self.lost_counter = 0

        return {}

    def step(self, data):
        """
        Verarbeitung eines einzelnen Frames.

        Wir holen uns die Position des verfolgten Fingers aus dem
        ``detector``-Signal, hängen sie an die Spur an und zeichnen die
        Spur als Linienzug ins Kamerabild. Ist keine Hand zu sehen,
        zählen wir nur den Verlust-Zähler hoch -- die Spur bleibt stehen.

        Parameters
        ----------
        data : dict
            Enthält unter anderem:

            - ``detector`` : erkannte Hände und Landmarken
            - ``config`` : Systemkonfiguration

        Returns
        -------
        dict
            ``galy`` : Visualisierungsobjekt mit der gezeichneten Spur.
        """
        galy = GALY()
        # Wir zeichnen die Spur auf einen eigenen Layer namens "trail".
        galy.layer("trail")
        # Die Fingerpunkte sind Zahlen zwischen 0 und 1, GALY zeichnet in Pixeln.
        # Also dieselbe Umrechnung wie im HandDetector: x mal Breite, y mal Hoehe.
        mapping = np.array([
            [self.webcam_width, 0.0, 0.0],
            [0.0, self.webcam_height, 0.0],
        ], dtype=np.float64)
        galy.set_layer_affine_mapping(mapping)

        detector = data.get('detector', {})
        hands = detector.get('hands', [])

        if hands:
            # Wir nehmen immer die erste erkannte Hand.
            hand = hands[0]
            landmarks = hand.get('landmarks', [])
            if len(landmarks) > self.finger_idx:
                # War die Hand laenger als max_lost weg, ist das eine NEUE Geste:
                # die eingefrorene Spur der vorigen Geste jetzt zuruecksetzen,
                # bevor die neue beginnt.
                if self.lost_counter > self.max_lost:
                    self.trail.clear()
                pos = (landmarks[self.finger_idx]['x'], landmarks[self.finger_idx]['y'])
                self.trail.append(pos)
                self.lost_counter = 0
            else:
                self.lost_counter += 1
        else:
            self.lost_counter += 1

        # Die Spur bleibt absichtlich stehen, wenn die Hand das Bild verlaesst --
        # so sieht der Aufnehmer den fertig gezeichneten Buchstaben und kann
        # beurteilen, ob die Aufnahme gut war. Zurueckgesetzt wird erst, wenn eine
        # neue Geste beginnt (siehe oben: Hand kommt nach >max_lost Frames zurueck).

        # Linien zwischen den gespeicherten Punkten zeichnen. Aeltere Abschnitte
        # werden dunkler (Blau -> Weiss), damit man die Richtung der Bewegung sieht.
        if len(self.trail) > 1:
            for i in range(1, len(self.trail)):
                start = self.trail[i-1]
                end = self.trail[i]
                alpha = i / len(self.trail)  # 0 = aeltester Punkt, 1 = neuester
                color = (int(255 * alpha), int(255 * alpha), 255)
                galy.line(start, end, color, 2)

        return {"galy": galy}

    def stop(self, data):
        """
        Wird beim Beenden aufgerufen. Wir halten keine externen Ressourcen
        (Dateien, Kameras, Modelle), deshalb gibt es hier nichts aufzuräumen.
        """
        pass