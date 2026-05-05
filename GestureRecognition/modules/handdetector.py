import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from SignalHub import GALY, bgr, get_nested_key, Module

mp_hand = mp.tasks.vision.HandLandmarksConnections


def draw_hand_landmarks(hand_landmarks, galy: GALY):
    lm = {
        "thumb":         {"color": bgr("#0000FF")},
        "index_finger":  {"color": bgr("#00FF00")},
        "middle_finger": {"color": bgr("#FF0000")},
        "ring_finger":   {"color": bgr("#00FFFF")},
        "pinky_finger":  {"color": bgr("#FF00FF")},
        "palm":          {"color": bgr("#C8C8C8")},
    }
    x = np.inf
    y = np.inf
    for key in lm.keys():
        pts = set()
        for conn in getattr(mp_hand, f"HAND_{key.upper()}_CONNECTIONS"):
            start = (hand_landmarks[conn.start].x,
                    hand_landmarks[conn.start].y)
            end = (hand_landmarks[conn.end].x,
                hand_landmarks[conn.end].y)
            x = min(x, start[0], end[0])
            y = min(y, start[1], end[1])
            galy.line(start, end, lm[key]["color"], 2)
            pts.update([conn.start, conn.end])
        for pt in pts:
            galy.circle((hand_landmarks[pt].x, hand_landmarks[pt].y), 5, (255,255,255), 1)
            galy.circle((hand_landmarks[pt].x, hand_landmarks[pt].y), 4, lm[key]["color"], -1)


class HandDetector(Module):
    """
    Modul zur Erkennung von Händen und deren Landmarken.

    Dieses Modul verwendet das MediaPipe Hand Landmarker Modell, um Hände
    in einem Kamerabild zu erkennen und deren Landmarken zu bestimmen.

    Ziel ist es, die Webcam-Bilder zu verarbeiten, eine Handdetektion
    durchzuführen und die erkannten Landmarken sowie eine Visualisierung
    an das Framework zurückzugeben.
    """

    def __init__(self, outputSignal="detector"):
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
        - ``webcam`` : aktuelles Kamerabild

        Zusätzlich muss ein **Output-Schema** definiert werden.

        Output Schema
        -------------
        Das Modul erzeugt ein Signal mit dem Namen ``detector``.

        Dieses Signal enthält das Ergebnis der Handdetektion, welches
        beispielsweise Informationen über erkannte Hände und Landmarken
        enthalten kann.

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
            inputSignals=["config", "webcam"],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="detector",
        )

    def start(self, data):
        """
        Initialisierung des Moduls.

        Diese Methode wird einmal beim Start des Moduls ausgeführt.
        Hier wird der MediaPipe HandLandmarker geladen.

        MediaPipe arbeitet intern in zwei Schritten:
        1. Palm detection: schnelle Handsuche im Bild
        2. Landmark regression: 21 Handpunkte für jede erkannte Hand

        Wir setzen ``num_hands=2``, damit ein bis zwei Hände erkannt werden.
        Die Confidence-Werte sind bewusst moderat eingestellt:
        - ``min_hand_detection_confidence=0.6``: zuverlässig, aber nicht zu streng,
          damit die Live-Erkennung noch schnell arbeitet.
        - ``min_hand_presence_confidence=0.5``: lässt schwächere Hände zu,
          aber schützt vor völligen Fehlalarmen.
        - ``min_tracking_confidence=0.5``: gute Balance zwischen Stabilität und Latenz.

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
        model_path = Path(__file__).resolve().parents[2] / "hand_landmarker.task"
        detection_confidence = get_nested_key(
            data,
            "config/hand_detection_confidence",
            0.6,
        )
        presence_confidence = get_nested_key(
            data,
            "config/hand_presence_confidence",
            0.5,
        )
        tracking_confidence = get_nested_key(
            data,
            "config/hand_tracking_confidence",
            0.5,
        )

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        return {}

    def step(self, data):
        """
        Verarbeitung eines einzelnen Frames.

        Hier wird jedes Bild von BGR nach RGB gewandelt, in ein MediaPipe
        ``mp.Image`` gepackt und dann an den Handlandmarker gegeben.

        Das Ergebnis liefert für jede erkannte Hand 21 normalisierte Punkte.
        Wir zeichnen diese Punkte mit der vorhandenen Funktion
        ``draw_hand_landmarks`` farbcodiert pro Finger.

        Parameters
        ----------
        data : dict
            Enthält unter anderem:

            - ``webcam`` : aktuelles Kamerabild
            - ``config`` : Systemkonfiguration

        Returns
        -------
        dict
            ``detector`` : erkannte Hände und Landmarken
            ``galy`` : Visualisierungsobjekt mit den gezeichneten Handverbindungen
        """
        galy = GALY()
        galy.layer("hands")

        image = get_nested_key(data, "webcam", None)
        if image is None:
            return {"detector": {"hands": []}, "galy": galy}

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        result = self.detector.detect(mp_image)
        hand_landmarks = getattr(result, "hand_landmarks", []) or []

        hands = []
        for hand_index, landmarks in enumerate(hand_landmarks[:2]):
            # MediaPipe liefert pro Hand 21 Landmarken.
            hand_data = {
                "id": hand_index,
                "landmarks": [
                    {"x": lm.x, "y": lm.y, "z": lm.z}
                    for lm in landmarks
                ],
            }
            hands.append(hand_data)
            draw_hand_landmarks(landmarks, galy)

        return {"detector": {"hands": hands}, "galy": galy}

    def stop(self, data):
        """
        Wird aufgerufen, wenn das Modul beendet wird.

        Hier geben wir das MediaPipe-Modell frei, damit keine
        GPU-/CPU-Ressourcen mehr blockiert werden.
        """
        if hasattr(self, "detector") and self.detector is not None:
            self.detector.close()
            self.detector = None
