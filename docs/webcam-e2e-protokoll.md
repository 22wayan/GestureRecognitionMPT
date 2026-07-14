# Webcam-End-to-End-Protokoll (echte Live-Aufnahme)

Die automatisierten Tests (`tests/test_end_to_end.py`, `tests/test_new_gesture.py`)
beweisen den kompletten **datengetriebenen** Pfad — ohne Webcam. Die
Aufgabenstellung verlangt zusätzlich einen echten Live-Durchlauf mit neuen Daten,
die per Kamera aufgenommen werden. Dieses Protokoll hält genau diesen Durchlauf fest.

## Durchführung

```bash
# 1) Neue Geste live aufnehmen (öffnet ein Kamera-Fenster)
python schnell_aufnahme.py testperson --symbols Dreieck --times 10

# 2) Modell neu trainieren (nimmt die neue Klasse in data/hmm.pkl auf)
python train.py

# 3) Live-Erkennung starten (Webcam)
python main.py --mode live
```

**Steuerung im Aufnahme-Fenster** (`schnell_aufnahme.py`):

| Taste | Wirkung |
|-------|---------|
| `LEERTASTE` | Aufnahme **starten**; nochmal drücken → Take **speichern** |
| `R` | aktuellen Take **verwerfen** |
| `BACKSPACE` | zuletzt gespeicherten Take löschen |
| `N` | dieses Zeichen überspringen |
| `Q` / `ESC` | beenden |

> [!WARNING]
> `python train.py` **überschreibt `data/hmm.pkl`** (gitignored, lokales
> Build-Artefakt). Für die Demo gewollt — die neue Klasse muss ins Modell.

> [!NOTE]
> Aufnahmen landen in `recordings/Dreieck/`. Auch `data_labeling()` schreibt nun
> direkt nach `recordings/<LABEL>/`, sodass `train.py` alle Aufnahmewege ohne
> Kopieren einliest. Ein Take wird nur gespeichert, wenn ≥ 20 Frames mit Hand
> erkannt wurden.

## Protokoll (Durchlauf vom 2026-07-14)

| Feld | Wert |
|------|------|
| Datum | 2026-07-14 |
| Rechner | MacBook Pro (Apple Silicon), macOS — *(ggf. präzisieren)* |
| Webcam-Index | 0 (`cv2.VideoCapture(0)`) |
| Name der neuen Geste | **Dreieck** |
| Zahl der Aufnahmen | 10 (10/10 bestehen die Trainings-Validierung, je `(48,3)`) |
| Speichern funktioniert? | ✅ ja — 10 Takes gespeichert |
| Verwerfen (`R`) funktioniert? | ☐ *vom Nutzer zu bestätigen* |
| Neue Klasse nach Training erkannt? | ✅ ja — live als „Dreieck" erkannt (Nutzer bestätigt); headless 10/10 |
| Beobachtete Trefferquote | live zuverlässig erkannt; headless-Vorabcheck 10/10; Gesamt-Test-Accuracy 0.912 (27 Klassen) |
| Bekannte Probleme | siehe unten (NaN-Kollaps bei wenigen Aufnahmen — behoben) |

## Beobachtungen / Findings

**NaN-Kollaps bei einer datenarmen Klasse (gefunden & behoben).**
Beim ersten Training mit der neuen Geste (8 Trainingssequenzen nach dem Split)
kollabierte das Klassenmodell bei `n_components=10` in `NaN`: ein Zustand bekam
kaum Daten, seine Varianz lief trotz `min_covar` in NaN, `score()` lieferte `-inf`
→ die Geste wurde **0/10** erkannt, obwohl sie „trainiert" war. Das ist derselbe
Mechanismus wie früher beim Buchstaben F, hier aber datenabhängig (ein bestimmter
Split kippt, ein anderer nicht — reine Messerschneide).

**Fix:** `HMMClassifier.fit` reduziert für eine kollabierende Klasse jetzt
automatisch die Zustandszahl, bis das Modell stabil ist (hier 10 → 9). Gut
besetzte Klassen (A–Z) behalten die volle Zustandszahl. Danach: Dreieck **10/10**
erkannt, Gesamt-Accuracy unverändert 0.912. Regressionstest:
`tests/test_hmm_nan_guard.py`.

**Relevanz für die Prüfung:** Genau dieses Szenario (Prüfer nimmt live eine neue
Geste mit wenigen Beispielen auf) hätte ohne den Fix still zu 0 % Recall geführt.

**Weitere Beobachtungen (vom Nutzer zu ergänzen):**

<!-- Licht / Abstand zur Kamera / Segmentierung (Geste zu früh/spät beendet?) / Verwechslungen -->

_____
