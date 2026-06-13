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

## Daten aufnehmen (Labeling)

Mit der Funktion `data_labeling(times, label)` in `GestureRecognition/labeling.py`
lassen sich eigene Trainingsdaten aufnehmen. Der Workflow ist so gebaut, dass ihn
auch eine Person ohne Vorwissen bedienen kann.

### Aufruf

```python
from GestureRecognition.labeling import data_labeling

# Nimmt 5 Aufnahmen fuer die Geste "A" auf
data_labeling(times=5, label="A")
```

- `times`: Anzahl der Aufnahmen, die fuer dieses Label **gespeichert** werden sollen.
- `label`: Name der Geste / Klasse (z. B. `"A"`).

Fuer mehrere Klassen wird die Funktion einfach nacheinander mit verschiedenen
Labels aufgerufen.

### Bedienschritte

1. Die Funktion startet pro Aufnahme `main.py` als Subprocess und zeichnet auf.
2. ENTER druecken, um eine Aufnahme zu starten.
3. Geste ausfuehren und das Aufnahme-Fenster schliessen, um die Aufnahme zu beenden.
4. Danach entscheiden:
   - `s` → Aufnahme **speichern**
   - `v` → Aufnahme **verwerfen** (wird wiederholt)
   - `a` → Vorgang **abbrechen**
5. Schritte wiederholen sich, bis `times` Aufnahmen gespeichert wurden.

Verworfene oder abgebrochene Aufnahmen hinterlassen keine Dateien, da zuerst in
eine temporaere Datei aufgenommen wird und diese nur beim Speichern in den
Datenordner verschoben wird.

### Datenstruktur und Ordnerorganisation

Gespeicherte Aufnahmen liegen nach Label getrennt im Ordner `data/`:

```
data/
├── A/
│   ├── recording_1.pkl
│   ├── recording_2.pkl
│   └── ...
├── B/
│   ├── recording_1.pkl
│   └── ...
└── ...
```

- Jedes Label bekommt einen eigenen Unterordner `data/{label}/`.
- Jede Aufnahme wird einzeln als `recording_{i}.pkl` gespeichert.
- Die Nummer `{i}` zaehlt fortlaufend hoch. Bestehende Aufnahmen werden nie
  ueberschrieben.

## Trainingsdatensatz bauen

Mit `dataset_building(output_path)` in `GestureRecognition/labeling.py` wird
aus den Rohaufnahmen unter `recordings/<label>/*.pkl` ein fertiger
Trainingsdatensatz für den `HMMClassifier` erzeugt.

### Aufruf

```python
from GestureRecognition.labeling import dataset_building

result = dataset_building("data/dataset.pkl")
```

Das Ergebnis ist sowohl als Rückgabewert als auch als Pickle-Datei unter
`output_path` verfügbar — ein Dict mit:

- `X_train`, `X_test`: konkatenierte Feature-Arrays (x, y, velocity)
- `y_train`, `y_test`: Klassenlabel pro Sequenz
- `lengths_train`, `lengths_test`: Länge jeder einzelnen Sequenz
- `classes`: sortierte Liste aller Klassenlabels

Direkt nutzbar für:

```python
classifier.fit(result["X_train"], result["y_train"], result["lengths_train"])
```

### Sequenz-Level-Split (Data Leakage vermeiden)

Jede Aufnahme ist eine ganze Geste, also eine Sequenz von Frames. Würde man
einzelne **Frames** zufällig auf Train/Test verteilen, könnten Frames
derselben Geste in beiden Sets landen. Das Modell hätte dann beim Testen
quasi schon Teile der Antwort gesehen — die Testgenauigkeit wäre künstlich
zu hoch. Deshalb wird auf Ebene ganzer **Aufnahmen** gesplittet: Jede
Sequenz landet komplett in genau einem der beiden Sets.

### Stratifizierung

`stratify=labels` (über `train_test_split`) sorgt dafür, dass Train- und
Test-Set für jede Klasse den gleichen Anteil an Aufnahmen enthalten. Bei
z. B. 10 Aufnahmen pro Klasse und `test_size=0.2` landen pro Klasse 2 im
Test-Set. Ohne Stratifizierung könnte eine Klasse bei wenigen Aufnahmen rein
zufällig komplett im Train- oder Test-Set landen.
