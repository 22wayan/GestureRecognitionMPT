# GestureRecognitionMPT

Ein kleines Projekt zur Gestenerkennung auf Basis von Hand-Landmarks.

Das Projekt demonstriert eine modulare Pipeline zur Erkennung von Gesten aus Webcam-Daten. Dafür werden Hand-Landmarks extrahiert und anschließend mit einem Hidden-Markov-Modell (HMM) klassifiziert.

## Pipeline

Die Verarbeitung erfolgt über mehrere Module:
```
Webcam → HandDetector → Preprocessor → HMMModule
```
- **HandDetector**
  Erkennt Hände im Kamerabild und extrahiert deren Landmarken.

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
