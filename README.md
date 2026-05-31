# GestureRecognitionMPT

MPT Projekt zur Erkennung von Gesten in Webcam-Daten.

Dafür werden Hand-Landmarks extrahiert und anschließend mit einem [Hidden-Markov-Modell](https://de.wikipedia.org/wiki/Hidden_Markov_Model) (HMM) klassifiziert.

Die Online-Dokumentation zur Bearbeitung des Projekts finden sie [hier](https://jaboll-ai.github.io/GestureRecognitionMPT).

## Pipeline

Die Verarbeitung erfolgt über mehrere Module:
```
Webcam → HandDetector → Preprocessor → HMMModule
```
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
