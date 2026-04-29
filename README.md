# GestureRecognitionMPT
MPT Projekt zur Erkennung von Gesten in Webcam-Daten.
Dafür werden Hand-Landmarks extrahiert und anschließend mit einem [Hidden-Markov-Modell](https://de.wikipedia.org/wiki/Hidden_Markov_Model) (HMM) klassifiziert.
Die Online-Dokumentation zur Bearbeitung des Projekts finden sie [hier](https://jaboll-ai.github.io/GestureRecognitionMPT).

## Pipeline
Die Verarbeitung erfolgt über mehrere Module:

- **HandDetector**
  Erkennt Hände im Kamerabild und extrahiert deren Landmarken. (optional: Darstellung der Hand)
- **Preprocessor**
  Sammelt und normalisiert Fingertrajektorien über mehrere Frames.
- **HMMModule**
  Klassifiziert Gesten mithilfe eines trainierten Hidden-Markov-Modells.
- **TrailMarker**
  Optionales Modul zur Visualisierung der Fingerbewegung.

<table>
<tr>
<td><img src="https://github.com/user-attachments/assets/f954735c-e8cb-4a82-9c38-4c748eb90dd4" width="250"></td>
<td><img src="https://github.com/user-attachments/assets/1ac89dba-d959-4a57-9ae3-a8db4629e1a3" width="250"></td>
<td><img src="https://github.com/user-attachments/assets/49a4a880-4def-4dc3-b807-c078870aa4f8" width="250"></td>
</tr>
<tr>
<td><img src="https://github.com/user-attachments/assets/c3947875-1300-414a-b939-96889eb490b6" width="250"></td>
<td><img src="https://github.com/user-attachments/assets/2e766180-9ecf-4434-a7a3-f0cf52b9b53e" width="250"></td>
<td><img src="https://github.com/user-attachments/assets/a85aa1e0-fe16-44f6-a180-c443b502a92b" width="250"></td>
</tr>
</table>

<img width="830" height="1430" alt="Dataset" src="https://github.com/user-attachments/assets/dd61fa9d-353a-46ed-adea-7a28238e1f9e" />

## Setup

### 1. Repo klonen
```bash
git clone https://github.com/22wayan/GestureRecognitionMPT.git
cd GestureRecognitionMPT
```

### 2. Dependencies installieren
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Kurz checken ob alles da ist:
```bash
python -c "import mediapipe, hmmlearn, numpy, cv2; print('OK')"
```

### 3. Recordings runterladen
```bash
curl -LO https://github.com/jaboll-ai/GestureRecognitionMPT/releases/download/recordings-v1/recordings.zip
unzip recordings.zip
```

### 4. MediaPipe-Modell runterladen
```bash
curl -LO https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

### 5. Webcam-Index einstellen
In `config.yml` den `deviceIndex` anpassen. Welcher Index passt, einfach testen:
```bash
python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    ok = cap.isOpened() and cap.read()[0]
    print(f'Index {i}: OK' if ok else f'Index {i}: -')
    cap.release()
"
```
Den ersten Index der `OK` ausgibt in der `config.yml` eintragen.

### 6. Replay-Modus testen
Auf macOS vorher den Qt-Pfad setzen, sonst crasht es:
```bash
export QT_QPA_PLATFORM_PLUGIN_PATH=$(python -c "import PyQt5, os; print(os.path.join(os.path.dirname(PyQt5.__file__), 'Qt5', 'plugins', 'platforms'))")
python main.py --mode replay --recorder.file recordings/A/A-1773050612.172112.pkl
```
Wenn ein Fenster mit bunten Punkten aufgeht, hat alles geklappt. Mit `Q` beenden.

### 7. Doku bauen (optional)
```bash
python -m sphinx -b html docs/source docs/build
open docs/build/index.html
```