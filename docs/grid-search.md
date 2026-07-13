# Grid Search: Warum `n_components = 10` die Wahl ist

Diese Seite dokumentiert, **wie wir die wichtigste Stellschraube des
HMM-Klassifikators gewählt haben** — nicht durch Raten, sondern durch
systematisches Messen (Grid Search). Ergebnis vorweg: **10 verborgene Zustände
pro Buchstabe** sind der beste Wert. Er ist jetzt der Standard in `train.py`.

Untersucht wird vor allem:

- **`n_components`** — die Anzahl der verborgenen Zustände pro Buchstaben-HMM
  (wie viele „Bewegungsphasen" eine Geste bekommt).

Die Kovarianz-Form (`diag` vs. `full`) wird in Abschnitt 4 kurz behandelt; der
Code selbst bleibt bei `diag` (Begründung dort).

---

## 1. Was ist Grid Search und Cross-Validation?

**Grid Search** heißt: man probiert mehrere Werte für einen Parameter
systematisch durch und misst, welcher am besten funktioniert.

**Cross-Validation (CV)** ist die faire Art, „am besten" zu messen. Statt nur
einmal in Training und Test zu teilen, wird der Datensatz in **5 gleich große
Teile (Folds)** zerlegt. Dann wird 5-mal trainiert: jedes Mal sind 4 Teile
Training und 1 Teil Test — und jeder Teil ist genau einmal der Test. Am Ende
mittelt man die 5 Genauigkeiten.

Vorteil: Das Ergebnis hängt **nicht vom Zufall einer einzelnen Aufteilung** ab.
Die **Standardabweichung** (std) über die 5 Folds verrät zusätzlich, wie
**stabil** eine Konfiguration ist — ein kleiner Wert heißt: verlässlich ähnliche
Ergebnisse, egal welche Aufnahmen gerade im Test landen.

Gemessen wurde auf **~1750 sauberen Sequenzen** (26 Klassen A–Z, 4 Personen),
5-fach-CV, `random_state=42`, `covariance_type="diag"`, `min_covar=0.03`.

---

## 2. Ergebnisse (n_components)

Mittlere CV-Genauigkeit (± Streuung über die 5 Folds):

| n_components | mittlere Accuracy | Streuung (std) | Bemerkung |
|---|---|---|---|
| 4  | 81,8 % | 0,012 | zu wenige Zustände → zu grob |
| 6  | 87,1 % | 0,014 | |
| 8  | 89,2 % | 0,018 | frühere Wahl |
| **10** | **91,1 %** | 0,026 | **bester Wert → neue Wahl** |
| 12 | 90,4 % | 0,040 | fällt wieder + unruhig → Overfitting |

**Ausgewählt: `n_components = 10`** (höchste mittlere Genauigkeit).

> **Zusätzlicher Beleg am echten Modell:** Trainiert man das ausgelieferte Modell
> (`python train.py`) mit `n_components=10`, steigt die Genauigkeit auf dem
> getrennten Test-Split von **90,3 % (bei 8) auf 93,1 % (bei 10)**. Grid-Search
> und echtes Training zeigen also in dieselbe Richtung.

---

## 3. Interpretation: die Anzahl der Zustände

Man sieht eine klare Kurve: Die Genauigkeit **steigt von 4 bis 10 Zuständen** und
**fällt bei 12 wieder ab**.

- **Zu wenige Zustände** (4): Das Modell ist zu grob, um die Form einer Geste
  abzubilden — nur ~82 %.
- **Sweet Spot** (10): Genug Zustände, um die einzelnen Bewegungsphasen eines
  Buchstabens sauber zu beschreiben — **91,1 %**.
- **Zu viele Zustände** (12): Das Modell fängt an, sich die Trainingsdaten zu
  „merken" statt zu verallgemeinern (**Overfitting**). Die Accuracy sinkt wieder
  und die Streuung steigt deutlich (0,040) — die einzelnen Folds schwanken dann
  stark (von 0,84 bis 0,95). Ein hoher, aber unruhiger Wert ist unzuverlässiger
  als ein etwas niedrigerer, stabiler.

Deshalb ist **10** die Wahl: höchster Mittelwert, und noch **vor** der
Overfitting-Klippe bei 12.

---

## 4. Kurz zur Kovarianz-Form (`diag` vs. `full`)

In einer einmaligen Zusatzmessung haben wir auch `full`-Kovarianz getestet.
Ergebnis: `full` mit `n_components=6` war mit ~91,2 % nur **hauchdünn** besser,
aber **numerisch instabil** — bei `n_components=8` bricht das Training ab:

```
ValueError: 'covars' must be symmetric, positive-definite
```

`full` hat viel mehr Parameter, für die pro Zustand zu wenige Daten da sind; die
Kovarianzmatrix „kollabiert". `diag` dagegen läuft über **alle** Zustandszahlen
stabil durch. Der winzige Genauigkeitsvorteil ist die Absturzgefahr nicht wert.

**Deshalb ist der Code bewusst auf `diag` festgelegt** — das hält die Grid-Search
einfach und robust.

---

## 5. Fazit / Designentscheidung

- **`covariance_type = "diag"`** — stabil; `full` ist fragil und kaum genauer.
- **`n_components = 10`** — höchste CV-Genauigkeit (91,1 %), belegt zusätzlich
  durch das echte Modell (Test-Accuracy 90,3 % → 93,1 %). Jetzt der Standard in
  `train.py`.

Die vorherige 8 war eine gute, etwas grobere Wahl; 10 ist messbar besser und
liegt noch vor dem Overfitting. **10 ist damit die belegte Wahl.**

---

## 6. Ergebnisse selbst reproduzieren

Die Grid-Search wurde bewusst **vereinfacht**: keine Datei-Reports, kein Plot,
nur die eigene stratifizierte Cross-Validation. Der Aufruf ist entsprechend
einfach — im Projekt-`.venv` (dort ist `hmmlearn` installiert):

```python
from GestureRecognition.labeling import clean_recordings
from GestureRecognition.grid_search import grid_search_n_components

# Alle sauberen Aufnahmen laden und flach machen.
data = clean_recordings("recordings")
sequences, labels = [], []
for label, seqs in data.items():
    for s in seqs:
        sequences.append(s); labels.append(label)

# grid_search_n_components filtert NaN-Sequenzen selbst heraus.
results, best = grid_search_n_components(sequences, labels)  # Standard: (4,6,8,10,12), diag, 5 Folds
for r in results:
    print(r["n_components"], round(r["mean_accuracy"], 4), round(r["std_accuracy"], 4))
print("Bestes n_components:", best["n_components"])   # -> 10
```

Oder direkt beim Training suchen und das beste Modell bauen lassen:

```bash
python train.py --grid-search
```

> **Hinweis:** Das HMM-Training enthält kleine Zufallsanteile — die Werte können
> sich von Lauf zu Lauf um ein paar Zehntel-Prozentpunkte unterscheiden. Die Kurve
> (Anstieg bis 10, Abfall bei 12) und die Wahl von 10 bleiben aber stabil.
