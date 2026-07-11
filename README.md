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

## Alphabet-Sammlung im Team (A–Z)

Damit das Team schnell und reibungslos Trainingsdaten sammeln kann, gibt es ein
geführtes Skript, das eine Person automatisch durch **alle 26 Buchstaben** führt
und pro Buchstabe **mehrere** Aufnahmen macht (Standard: 15). Die Aufnahmen
landen direkt in `recordings/<Buchstabe>/` — also genau dort, wo
`dataset_building()` liest.

> **Vor dem ersten Lauf** unbedingt den Webcam-Index setzen (siehe
> [Schritt 5](#5-webcam-index-einstellen)), sonst bleibt das Fenster schwarz.
> `deviceIndex` ist pro Rechner unterschiedlich — diese lokale Einstellung
> **nicht** committen. Auf macOS außerdem vorher den Qt-Pfad setzen, sonst
> crasht das Fenster:
> ```bash
> export QT_QPA_PLATFORM_PLUGIN_PATH=$(python -c "import PyQt5, os; print(os.path.join(os.path.dirname(PyQt5.__file__), 'Qt5', 'plugins', 'platforms'))")
> ```

### Aufruf

```bash
python collect_alphabet.py --person arian
# oder ohne Argument, dann wird der Name abgefragt:
python collect_alphabet.py
# Anzahl Aufnahmen pro Buchstabe anpassen (Standard 15):
python collect_alphabet.py --person arian --times 15
```

- Pro Buchstabe: ENTER → Geste in die Luft malen → Fenster schließen →
  `s`/`v`/`a` (speichern / verwerfen / abbrechen).
- Die gezeichnete Spur **bleibt stehen**, wenn die Hand das Bild verlässt — so
  siehst du den fertigen Buchstaben und erkennst sofort, ob die Aufnahme sauber
  war (ein wilder Querstrich = Tracking-Sprung = lieber `v`).
- **Resume:** Bereits aufgenommene Takes dieser Person werden übersprungen. Der
  Durchlauf kann jederzeit abgebrochen und später fortgesetzt werden — nichts
  wird überschrieben.

### Dateibenennung

Jede Aufnahme heißt `recordings/<Buchstabe>/<Buchstabe>-<person>-<n>.pkl`
(z. B. `recordings/A/A-arian-1.pkl`). So bleibt nachvollziehbar, wer welche
Aufnahmen gemacht hat, und nichts wird überschrieben.

### Aufnahmen prüfen & aussortieren

Nicht jede Aufnahme ist brauchbar (zu kurz, Tracking-Sprung, Hand kurz
verloren). `review_recordings.py` prüft alle eigenen Aufnahmen mit **denselben
Kriterien wie `dataset_building`** und verschiebt schlechte optional nach
`recordings_rejected/` (löscht nichts):

```bash
python review_recordings.py --person arian              # nur Bericht
python review_recordings.py --person arian --quarantine # schlechte aussortieren
```

Workflow bis z. B. 15 gute Aufnahmen pro Buchstabe:

```bash
python collect_alphabet.py --person arian               # fehlende nachnehmen
python review_recordings.py --person arian --quarantine # prüfen & aussortieren
# wiederholen, bis "fehlt 0"
```

### Danach: Datensatz bauen

Sind alle Personen durch, einmalig den Trainingsdatensatz erzeugen:

```bash
python -c "from GestureRecognition.labeling import dataset_building; dataset_building('data/dataset.pkl')"
```

Details dazu im nächsten Abschnitt.

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

## Datensatz visualisieren

Mit `visualize_dataset()` in `GestureRecognition/visualization.py` lässt sich
der bereinigte Datensatz (über `clean_recordings`) visuell prüfen. Die
Funktion erzeugt drei PNG-Dateien unter `plots/`, jede Klasse hat dabei über
alle Plots hinweg dieselbe Farbe.

```python
from GestureRecognition.visualization import visualize_dataset

visualize_dataset()
```

### `trajectories_per_class.png`

Überlagert die (x, y)-Pfade aller Aufnahmen einer Klasse, der Startpunkt ist
als Punkt markiert.

**Interpretation:** Die meisten Klassen (z. B. A, B, E, M, S, X) zeigen eine
klar wiedererkennbare, der jeweiligen Buchstabenform entsprechende
Trajektorie mit guter Überdeckung der Aufnahmen — die Geste wird also
konsistent ausgeführt. Bei einigen Klassen (z. B. I, V, T) ist die Form sehr
einfach/linear, was sie potenziell schwerer von ähnlichen Klassen
unterscheidbar macht. Einzelne abweichende Linien innerhalb einer Klasse
(z. B. vereinzelt bei K oder X) sind Ausreißer-Aufnahmen, die im Training zu
Rauschen führen können.

### `sequence_length_histogram.png`

Histogramm der Sequenzlängen (Anzahl Frames nach Bereinigung) pro Klasse.

**Interpretation:** Die meisten Klassen liegen im Bereich von ca. 50–125
Frames mit einer relativ engen, eingipfligen Verteilung — die Geste wird
zeitlich konsistent ausgeführt. Klassen mit breiter gestreuten oder
mehrgipfligen Histogrammen (z. B. C, K, X) deuten auf unterschiedlich schnell
ausgeführte Wiederholungen derselben Geste hin, was die HMM-Zustände stärker
beanspruchen kann.

### `velocity_profiles.png`

Geschwindigkeit pro Frame über die normalisierte Zeit (0–1) für jede
Aufnahme, mit Mittelwert-Trajektorie (schwarz) pro Klasse.

**Interpretation:** Viele Klassen zeigen ein wiederkehrendes
Mehrfach-Maxima-Muster (z. B. M, W, X), das zu den mehreren "Spitzen" der
jeweiligen Buchstabenform passt — die Mittelwertkurve folgt den einzelnen
Aufnahmen gut, was auf konsistentes Bewegungstempo hindeutet. Aufnahmen, deren
Geschwindigkeitskurve stark vom Mittelwert abweicht (sichtbar als einzelne
"ausreißende" farbige Linien, z. B. bei H oder K), sind Kandidaten für
Tracking-Probleme oder untypisch ausgeführte Gesten.

### Schlussfolgerungen für die Datenqualität

Insgesamt sind die Trajektorien pro Klasse konsistent genug für ein
HMM-Training. Klassen mit hoher Streuung in Länge und Geschwindigkeit (z. B.
C, K, X) sollten bei schlechter Klassifikationsleistung zuerst überprüft
werden — entweder durch zusätzliche, konsistentere Aufnahmen oder durch
Entfernen einzelner Ausreißer-Sequenzen.

## Ergebnisse

Wie gut erkennt das trainierte Modell die Buchstaben-Gesten? Gemessen mit
`evaluate_classifier()` auf dem eigenen A–Z-Datensatz:

| Test | Bedeutung | Accuracy |
|---|---|---|
| **Standard** | bekannte Personen, neue Aufnahmen | **~79 %** |
| **Neue Person** | eine Person komplett aus dem Training rausgehalten | **~36 %** |

Die zweite Zahl ist die ehrliche Erwartung für die Prüfung, in der der Prüfer
live als **völlig neue Person** Gesten aufnimmt.

<img src="docs/source/_static/confusion_matrix.png" width="520">

Die Confusion Matrix zeigt, **welche** Buchstaben verwechselt werden (Zeile =
wirklich gemacht, Spalte = vom Modell geraten, Diagonale = richtig). Verwechselt
werden vor allem Buchstaben mit ähnlicher Bewegung (z. B. P↔F, N↔P, E↔S).

➡️ Ausführliche Erklärung, Interpretation und Reproduktions-Anleitung:
**[docs/ergebnisse.md](docs/ergebnisse.md)**
