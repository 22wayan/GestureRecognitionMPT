# Live-Modus: Gesten in Echtzeit erkennen

Diese Seite erklärt den **Live-Modus** – den finalen Integrationstest, bei dem
die komplette Pipeline von der Webcam läuft und eine gemalte Geste **in Echtzeit**
klassifiziert wird. Sie ist bewusst für Einsteiger geschrieben.

> In der Prüfung nimmt der Prüfer **live** neue Gesten auf. Der Live-Modus ist
> genau die Situation, die dann getestet wird.

---

## 1. Was macht der Live-Modus?

Alle Module arbeiten Frame für Frame zusammen (die „Pipeline"):

```
Webcam  →  HandDetector  →  Preprocessor  →  HMMModule
 Bild        Hand +          sammelt die       vergleicht die
 holen       Fingerspitze    Bewegung &        Bewegung mit dem
             finden          normalisiert      trainierten Modell
                                               → zeigt den Buchstaben
```

- **Webcam**: liefert das Kamerabild.
- **HandDetector**: findet die Hand und ihre Landmarken (Gelenkpunkte).
- **Preprocessor**: verfolgt die **Zeigefingerspitze** und sammelt die Bewegung.
- **HMMModule**: vergleicht die fertige Bewegung mit allen 26 Buchstaben-Modellen
  und blendet den wahrscheinlichsten oben links ins Kamerabild ein.

---

## 2. Live-Modus starten

**Voraussetzung:** Es muss ein trainiertes Modell unter `data/hmm.pkl` liegen.
Falls nicht, vorher einmal `python train.py` ausführen (sonst erscheint als Label
dauerhaft `...`, weil das Modell fehlt).

```bash
# macOS: zuerst den Qt-Pfad setzen, sonst crasht das Anzeigefenster
export QT_QPA_PLATFORM_PLUGIN_PATH=$(python -c "import PyQt5, os; print(os.path.join(os.path.dirname(PyQt5.__file__), 'Qt5', 'plugins', 'platforms'))")

# Live-Modus starten (Webcam läuft, sobald der Modus nicht "replay" ist)
python main.py --mode live
```

Vorher in `config.yml` den richtigen `webcam.deviceIndex` einstellen (siehe
README, Schritt 5). Beenden mit der Taste **`Q`**.

> **Tipp:** Der allererste Start dauert 15–40 s, weil Python große Bibliotheken
> (mediapipe, sklearn, hmmlearn) lädt und einmalig kompiliert. Nicht mit `Ctrl+C`
> abbrechen – einfach warten, bis das Kamerafenster aufgeht.

---

## 3. So läuft eine Erkennung ab

1. Hand ins Bild halten – die Fingerspitze wird verfolgt (rote/weiße Spur).
2. Einen Buchstaben **in die Luft malen**.
3. Am Ende **kurz stoppen** oder die **Hand aus dem Bild nehmen**.
4. Oben links im Kamerabild erscheint das Ergebnis:
   - **Label**: der erkannte Buchstabe (oder `?`, siehe unten)
   - **Score**: wie gut die Bewegung zum besten Modell passt (höher = besser)
   - **Margin**: der Abstand zum zweitbesten Buchstaben (höher = eindeutiger)

Das Label **bleibt stehen**, bis die nächste Geste kommt – so kann man es in Ruhe
ablesen.

**Wann wird klassifiziert?** Erst **wenn die Geste zu Ende ist** – also wenn die
Hand deutlich langsamer wird *oder* aus dem Bild verschwindet. Während des Malens
wird das alte Label weiter angezeigt. Das ist Absicht: Das Modell braucht die
**komplette** Bewegung, um zu entscheiden.

---

## 4. Edge Cases (Sonderfälle) und wie sich das System verhält

Diese Fälle sind im Code bewusst behandelt:

| Sonderfall | Verhalten | Wo im Code |
|---|---|---|
| **Hand verlässt das Bild** | Nach ca. 10 verlorenen Frames (~0,3 s bei 30 fps) gilt die Geste als beendet und wird klassifiziert. | `max_lost` in `config.yml` / [preprocessor.py](../GestureRecognition/modules/preprocessor.py) |
| **Mehrdeutige Geste** (zwei Buchstaben fast gleich wahrscheinlich) | Das Modell zeigt **`?`**, wenn der `Margin` zu klein ist (< 0,1). | [hiddenmarkov.py](../GestureRecognition/modules/hiddenmarkov.py) |
| **Sehr unsichere Geste** | Ebenfalls **`?`**, wenn der `Score` unter dem Schwellwert (−20) liegt. | `hiddenmarkov.py` |
| **Zu kurze / winzige Bewegung** | Wird verworfen (weniger als `min_steps` = 15 Punkte), keine Ausgabe. | `preprocessor.py` |
| **Mehrere Buchstaben nacheinander** | Nach jeder erkannten Geste wird der Puffer geleert, damit die nächste Geste sauber (ohne Reste) startet. | `preprocessor.py` |
| **Kein Modell vorhanden** | Label bleibt `...`, das System läuft aber weiter. → `python train.py` ausführen. | `hiddenmarkov.py` |

Die **`?`-Anzeige ist ein Feature, kein Fehler**: Es ist besser, „weiß nicht"
zu sagen, als bei einer unklaren Bewegung einen falschen Buchstaben zu behaupten.

---

## 5. Latenz (Verzögerung)

Es gibt zwei Arten von Verzögerung:

1. **Pro Frame** (Bild holen → Hand finden → puffern): Gemessen **~33 ms pro
   Frame** (≈ 30 Bilder/Sekunde, Anzeige „Speed: Fast (33 ms)"). Der teuerste
   Schritt ist die MediaPipe-Handerkennung.
2. **Bis das Ergebnis erscheint**: Das Label kommt erst **nach dem Ende der
   Geste**. Wer die Geste durch Herausnehmen der Hand beendet, wartet ca. die
   `max_lost`-Zeit (~10 Frames ≈ 0,3 s). Wer am Ende nur kurz stoppt, bekommt das
   Ergebnis fast sofort.

---

## 6. Beobachtungen aus dem echten Live-Lauf

**Lauf am 2026-07-11 (macOS, interne Webcam):**

- **Anzeige:** Das Label erscheint zuverlässig oben links im Kamerabild und
  **bleibt stehen**, bis die nächste Geste kommt (z. B. `Label: F / Score: 2.679
  / Margin: 0.303`).
- **Geschwindigkeit:** flüssig, konstant „Speed: Fast (33 ms)" (~30 fps).
- **Stabilität:** lief über mehrere hundert Frames (Scan-Zähler > 600) ohne
  Absturz, keine Fehlermeldung im Terminal. Speicherverbrauch ~300–500 MB.
- **Erkennung:** Einzeln und klar gemalte Buchstaben werden erkannt; der Score
  liegt bei guten Gesten deutlich über 0, bei unklaren Bewegungen sinkt der
  Margin und es erscheint `?`.
- **Grenzen:** Die Erkennung ist nicht perfekt – siehe nächster Abschnitt und die
  [Ergebnisse](ergebnisse.md). Für eine **völlig neue Person** (Prüfer) ist die
  Genauigkeit deutlich niedriger als für bekannte Personen.

---

## 7. Was der Integrationstest gefunden und behoben hat

Der erste End-to-End-Test hat einen echten Fehler aufgedeckt: Kamera und Tracking
liefen, aber es erschien **nie ein Buchstabe**. Ursachen (alle behoben):

1. **Invertierte Hysterese** in `config.yml`: Die Start-Schwelle
   (`min_speed_corner`) lag *unter* der Stopp-Schwelle (`reset_speed_corner`).
   Dadurch wurde jede Geste sofort abgewürgt und in winzige Fetzen zerhackt – das
   Modell sah nur Teil-Gesten und gab fast immer `?` aus. **Fix:** Start-Schwelle
   über die Stopp-Schwelle gesetzt (0.005 > 0.001), sodass die Geste erst bei
   echter Pause/Hand-Weg endet und als *ganze* Trajektorie beim Modell ankommt.
2. **`margin_threshold` zu streng** (0.5): verwarf selbst korrekte Top-Tipps als
   `?`. **Fix:** auf 0.1 gesenkt.
3. **Anzeige-Fehler:** Das HMM-Modul zeichnete in eine *eigene* Fläche, die als
   separates, schwarzes Fenster geöffnet wurde und leer blieb. **Fix:** Das Label
   wird jetzt als Ebene **direkt auf das Kamerabild** gezeichnet (wie die
   Handpunkte und die Spur) und bleibt zwischen zwei Gesten stehen.

In einem Offline-Test der Live-Segmentierung stieg dadurch die Top-1-Erkennung
von **12 % auf 69 %** (bei einer dem Training bekannten Person). Ein
Regressionstest ([tests/test_live_segmentation.py](../tests/test_live_segmentation.py))
sichert ab, dass die Hysterese nicht wieder invertiert wird.

---

## 8. Kurz-Checkliste für die Prüfung

- [ ] `data/hmm.pkl` existiert (sonst `python train.py`)
- [ ] Richtiger `webcam.deviceIndex` in `config.yml`
- [ ] macOS: `QT_QPA_PLATFORM_PLUGIN_PATH` gesetzt
- [ ] `python main.py --mode live` startet ein Fenster mit Kamerabild
- [ ] Ein gemalter Buchstabe erzeugt oben links ein Label
- [ ] Beenden mit `Q`
