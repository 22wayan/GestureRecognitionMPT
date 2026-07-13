# Designentscheidungen: Warum ist was so eingestellt?

Diese Seite sammelt alle wichtigen Entscheidungen an einem Ort: welche
Features wir benutzen, welche Hyperparameter das Modell hat und welche
Schwellenwerte im Live-Modus gelten — jeweils mit einer kurzen Begründung.
Alle Werte stimmen mit dem aktuellen Code überein (`config.yml`,
`GestureRecognition/labeling.py`, `train.py`).

## 1. Features: Was bekommt das Modell überhaupt zu sehen?

Das Modell sieht keine Bilder. Aus jeder Geste machen wir eine Liste von
Zahlen — pro Punkt der Fingerspur genau **3 Werte**:

| Feature | Bedeutung |
|---|---|
| `x`, `y` | Position der Zeigefingerspitze (Landmark 8) |
| `velocity` | Abstand zum vorherigen Punkt (wie schnell bewegt sich der Finger) |

**Warum die Zeigefingerspitze (`finger_idx: 8`)?** Man malt Buchstaben
natürlich mit dem Zeigefinger — die Spitze ist der Punkt, der die Form
am deutlichsten nachzeichnet.

**Warum Geschwindigkeit dazu?** Zwei Buchstaben können ähnliche Formen
haben, aber ein unterschiedliches Tempo-Muster (z. B. wo man abbremst,
weil eine Ecke kommt). Die Geschwindigkeit gibt dem Modell diese
Zusatzinformation.

Die Umwandlung passiert in **einer gemeinsamen Funktion**
(`labeling._to_features`), die Training **und** Live-Modus benutzen.
So können die beiden nie auseinanderlaufen — ein Format-Unterschied
zwischen Training und Live war früher ein echter Bug (Score `-inf`,
live kam immer nur „?").

## 2. Resampling: jede Geste bekommt gleich viele Punkte

**Problem:** Jeder malt unterschiedlich schnell. Eine schnelle Person
erzeugt ~26 Punkte pro Buchstabe, eine langsame ~70. Für das Modell sahen
ein schnelles und ein langsames „A" deshalb verschieden aus.

**Lösung:** Jede Geste wird per linearer Interpolation (`np.interp`) auf
genau **48 Punkte** gebracht (`RESAMPLE_LENGTH = 48` in `labeling.py`) —
wie ein Bild, das man auf eine feste Größe zieht. Danach ist das Tempo egal.

**Wirkung (gemessen):** Standard-Accuracy stieg von ~72 % auf ~90 %.
Der Wert 48 kam aus einem Vergleich mehrerer Längen (32/48/64) — 48 war
der beste Kompromiss aus Genauigkeit und Rechenzeit.

## 3. Normalisierung: egal wo und wie groß gemalt wird

Jede Spur wird zentriert (Mittelpunkt abziehen) und auf den
**Einheitskreis** skaliert (durch den größten Abstand zum Mittelpunkt
teilen). Damit ist es egal, ob jemand oben links klein oder in der Mitte
groß malt — die Form zählt, nicht die Position oder Größe.

## 4. HMM-Hyperparameter

Trainiert wird **ein eigenes GaussianHMM pro Buchstabe** (26 Modelle).
Beim Erkennen gewinnt das Modell mit der höchsten Log-Likelihood.

| Parameter | Wert | Warum |
|---|---|---|
| `n_components` | **10** | Anzahl der verborgenen Zustände (≈ Phasen einer Geste). Per Grid-Search mit Cross-Validation als bester Wert belegt — Details in [grid-search.md](grid-search.md). |
| `covariance_type` | **diag** | Deutlich weniger Parameter als `full` → stabiler bei unseren Datenmengen. Auch das wurde verglichen (siehe grid-search.md). |
| `min_covar` | **0.03** | Untergrenze für die Varianz jedes Zustands. Ohne sie kann ein Zustand auf einen Punkt „kollabieren" (Varianz → 0 → NaN) und das ganze Klassenmodell wird unbrauchbar — genau das passierte beim Buchstaben F (0 % Erkennung). Mit 0.03: F wieder 100 %, Gesamt-Accuracy 88 % → 90 %. |
| `n_iter` / `tol` | 100 / 1e-2 | Standardwerte, reichen für Konvergenz bei unseren kurzen Sequenzen. |
| `random_state` | 42 | Reproduzierbarkeit — gleicher Lauf, gleiches Ergebnis. |

## 5. Live-Segmentierung: Wann beginnt und endet eine Geste?

Im Live-Modus muss das Programm selbst merken, wann jemand anfängt und
aufhört zu malen. Das läuft über die Geschwindigkeit der Fingerspitze
(Werte in `config.yml`):

| Parameter | Wert | Warum |
|---|---|---|
| `min_speed_corner` | 0.005 | **Start-Schwelle:** ab dieser Bewegung beginnt das Sammeln. |
| `reset_speed_corner` | 0.001 | **Stopp-Schwelle:** erst wenn die Hand fast steht, endet die Geste. Wichtig: Start-Schwelle > Stopp-Schwelle (Hysterese) — sonst wird die Geste bei jeder kleinen Tempo-Delle zerhackt. |
| `stop_hold` | 4 | So viele langsame Frames **hintereinander** müssen kommen, bevor die Geste endet („Entprellen"). Ein kurzes Abbremsen an einer Ecke (M, W, Z!) beendet den Buchstaben so nicht mehr mittendrin. |
| `buffer_size` | 250 | Maximal gesammelte Punkte (~8 Sekunden bei 30 fps) — genug für groß/langsam gemalte Buchstaben. |
| `max_lost` | 10 | Nach so vielen Frames ohne Hand gilt die Geste als beendet. |
| `min_steps` | 15 | Kürzere Spuren werden verworfen (das ist Zittern, kein Buchstabe). |

## 6. Live-Entscheidung: Wann sagt das Modell lieber „?"

Das beste Klassenmodell gewinnt — aber nur, wenn es sich „sicher genug" ist:

| Parameter | Wert | Bedeutung |
|---|---|---|
| `score_threshold` | −20.0 | Ist selbst der beste Score schlechter, passt kein Buchstabe → „?" |
| `margin_threshold` | 0.1 | Ist der Abstand zwischen bestem und zweitbestem Buchstaben kleiner, ist die Sache mehrdeutig → „?" (0.5 war zu streng und hat korrekte Tipps verworfen) |

## 7. Datenbereinigung und Train/Test-Split

- **`min_length = 15`:** Aufnahmen mit weniger gültigen Frames fliegen raus.
- **`max_jump = 0.15`:** Springt der Finger zwischen zwei Frames weiter als
  15 % der Bildbreite, ist das ein Tracking-Fehler → Aufnahme fliegt raus
  (echte Bewegungen liegen weit darunter, Maximum ~0.11).
- **Split auf Sequenz-Ebene** (ganze Aufnahmen, nie einzelne Frames) mit
  `test_size = 0.2` und Stratifizierung — verhindert Data Leakage,
  Details im README-Abschnitt „Trainingsdatensatz bauen".

## 8. Kurzfazit

Die zwei wichtigsten Entscheidungen waren **Resampling auf feste Länge**
(Tempo-Unterschiede zwischen Personen verschwinden) und **`min_covar`**
(kein stilles Kaputtgehen einzelner Klassenmodelle). Zusammen mit mehr
Trainingsdaten von vier Personen kam die Standard-Accuracy so von ~72 %
auf ~90 %.
