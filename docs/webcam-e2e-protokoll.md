# Webcam-End-to-End-Protokoll (echte Live-Aufnahme)

Die automatisierten Tests (`tests/test_end_to_end.py`, `tests/test_new_gesture.py`)
beweisen den kompletten **datengetriebenen** Pfad — ohne Webcam. Die
Aufgabenstellung verlangt zusätzlich einen echten Live-Durchlauf mit neuen Daten,
die per Kamera aufgenommen werden. Dieses Protokoll hält genau diesen Durchlauf fest.

## Durchführung

```bash
# 1) Neue Geste live aufnehmen (öffnet ein Kamera-Fenster)
python schnell_aufnahme.py testperson --symbols TESTGESTE --times 10

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
> `python train.py` **überschreibt `data/hmm.pkl`**. Das ist für die Demo gewollt
> (die neue Klasse muss ins Modell). Vorher ggf. sichern, wenn das alte Modell
> gebraucht wird: `cp data/hmm.pkl data/hmm.backup.pkl`.

> [!NOTE]
> Aufnahmen landen in `recordings/TESTGESTE/`. Nur was in `recordings/<LABEL>/`
> liegt, wird von `train.py` gelernt (`data_labeling()` schreibt abweichend nach
> `data/{label}/` — für diese Demo also `schnell_aufnahme.py` verwenden).
> Ein Take wird nur gespeichert, wenn ≥ 20 Frames mit Hand erkannt wurden.

## Protokoll (auszufüllen)

| Feld | Wert |
|------|------|
| Datum | _____ (z. B. 2026-07-14) |
| Rechner | _____ (Modell / OS) |
| Webcam-Index | _____ (`schnell_aufnahme.py` nutzt `cv2.VideoCapture(0)` → i. d. R. 0) |
| Name der neuen Geste | TESTGESTE |
| Zahl der Aufnahmen | _____ (Ziel: 10) |
| Speichern funktioniert? | ☐ ja ☐ nein — Anmerkung: _____ |
| Verwerfen (`R`) funktioniert? | ☐ ja ☐ nein — Anmerkung: _____ |
| Neue Klasse nach Training erkannt? | ☐ ja ☐ nein |
| Beobachtete Trefferquote (grob) | _____ % (z. B. „8 von 10 Versuchen") |
| Bekannte Probleme | _____ (Licht / Abstand / Segmentierung / Verwechslungen) |

## Beobachtungen / Notizen

<!-- Freitext: Was lief gut, was war fragil, was würde man beim nächsten Mal anders machen? -->

_____
