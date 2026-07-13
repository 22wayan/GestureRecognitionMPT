# Grid Search: Welche Modell-Konfiguration ist die beste?

Diese Seite dokumentiert, **wie wir die Hyperparameter des HMM-Klassifikators
gewählt haben** — nicht durch Raten, sondern durch systematisches Ausprobieren
und Messen (Grid Search). Sie richtet sich an Einsteiger: jeder Begriff wird
kurz erklärt.

Zwei Stellschrauben werden untersucht:

- **`n_components`** — die Anzahl der verborgenen Zustände pro Buchstaben-HMM
  (wie viele „Phasen" eine Geste hat).
- **`covariance_type`** — die Form der Kovarianzmatrix: `diag` (nimmt an, dass
  x, y und Geschwindigkeit unabhängig sind) oder `full` (erlaubt Zusammenhänge
  zwischen ihnen, hat aber viel mehr Parameter).

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
Zusätzlich verrät die **Standardabweichung** (std) über die 5 Folds, wie
**stabil** eine Konfiguration ist — ein kleiner Wert bedeutet: das Modell liefert
zuverlässig ähnliche Ergebnisse, egal welche Aufnahmen gerade im Test landen.

Gemessen wurde auf **1749 sauberen Sequenzen** (26 Klassen A–Z, 4 Personen),
5-fach-CV, `random_state=42`, `min_covar=0.03` (wie im echten Modell).

---

## 2. Ergebnisse

Mittlere CV-Genauigkeit (± Standardabweichung über die 5 Folds):

| covariance_type | n_components | mittlere Accuracy | Streuung (std) | Bemerkung |
|---|---|---|---|---|
| diag | 4  | 78,3 % | 0,035 | zu wenige Zustände |
| diag | 6  | 86,3 % | 0,035 | |
| diag | **8** | **89,2 %** | 0,016 | **aktuelle Wahl im Modell** |
| diag | 10 | **90,8 %** | **0,009** | bester `diag`-Wert, am stabilsten |
| diag | 12 | 88,4 % | 0,038 | fällt wieder → Overfitting |
| full | 6  | 91,2 % | 0,008 | knapp bester Wert überhaupt … |
| full | 8  | — | — | **Fit gescheitert** (Kovarianz nicht mehr positiv-definit) |

---

## 3. Interpretation: die Anzahl der Zustände (`n_components`)

Man sieht eine klare Kurve: Die Genauigkeit **steigt von 4 bis 10 Zuständen** und
**fällt bei 12 wieder ab**.

- **Zu wenige Zustände** (4): Das Modell ist zu grob, um die Form einer Geste
  abzubilden — nur 78 %.
- **Sweet Spot** (8–10): Genug Zustände, um die einzelnen Bewegungsphasen eines
  Buchstabens zu beschreiben — ~89–91 %.
- **Zu viele Zustände** (12): Das Modell fängt an, sich die Trainingsdaten zu
  „merken" statt zu verallgemeinern (**Overfitting**) — die Accuracy sinkt wieder
  und die Streuung steigt (0,038).

Der beste `diag`-Wert ist **`n_components=10`** (90,8 %) — mit der **kleinsten
Streuung** (0,009), also am stabilsten. Die aktuell im Modell gesetzte **8**
liegt mit 89,2 % nur knapp darunter und ist etwas einfacher (weniger Parameter).

---

## 4. Interpretation: die Kovarianz-Form (`diag` vs. `full`)

`full` kann theoretisch mehr abbilden (Zusammenhänge zwischen x, y und
Geschwindigkeit) und erreicht mit `n_components=6` auch den **knapp höchsten
Wert (91,2 %)**. **Aber:** `full` hat viel mehr Parameter und ist dadurch
**numerisch instabil** — bei `n_components=8` **bricht das Training ab**:

```
ValueError: 'covars' must be symmetric, positive-definite
```

Die Kovarianzmatrix „kollabiert", weil pro Zustand zu wenige Daten für so viele
Parameter da sind. `diag` dagegen läuft über **alle** getesteten Zustandszahlen
stabil durch.

**Trade-off:** `full` bringt gegenüber `diag` nur ~0,4 Prozentpunkte (91,2 % vs.
90,8 %), riskiert aber Abstürze. Der kleine Gewinn ist die Instabilität nicht
wert.

---

## 5. Fazit / Designentscheidung

- **`covariance_type = "diag"`** — stabil über alle Zustandszahlen; `full` ist
  fragil (bricht bei `n_components=8` ab) und kaum genauer.
- **`n_components`** liegt sinnvoll bei **8–10**. Die aktuelle **8** ist eine gute,
  etwas einfachere Wahl; **10** wäre laut CV marginal besser (90,8 % statt 89,2 %)
  und am stabilsten — ein leichter, gefahrloser Verbesserungshebel.

Diese Messung bestätigt also die Grundwahl (`diag`, ~8 Zustände) und zeigt
gleichzeitig einen konkreten kleinen Optimierungsschritt auf (`n_components=10`).

---

## 6. Ergebnisse selbst reproduzieren

Im Projekt-`.venv` (dort ist `hmmlearn` installiert):

```python
import numpy as np
from GestureRecognition.labeling import clean_recordings
from GestureRecognition.grid_search import _evaluate_configuration

# Alle sauberen Sequenzen laden (NaN-Sequenzen ueberspringen).
data = clean_recordings("recordings")
sequences, labels = [], []
for label, seqs in data.items():
    for s in seqs:
        s = np.asarray(s, dtype=float)
        if np.isfinite(s).all():
            sequences.append(s); labels.append(label)

# n_components durchprobieren (covariance_type="diag" ist stabil).
for n in (4, 6, 8, 10, 12):
    r = _evaluate_configuration(sequences, labels, n_components=n,
                                covariance_type="diag", n_splits=5, random_state=42)
    print(n, round(r["mean_accuracy"], 4), round(r["std_accuracy"], 4))
```

> **Hinweis 1:** Das HMM-Training enthält kleine Zufallsanteile — die Werte
> können sich von Lauf zu Lauf um ein paar Zehntel-Prozentpunkte unterscheiden.
>
> **Hinweis 2:** `python train.py --grid-search` nutzt die eingebaute Suche, aber
> mit dem engeren Standardbereich `n_components = 2…6`. Für den vollen Bereich
> (inkl. 8, 10, 12) das Snippet oben verwenden.
>
> **Hinweis 3:** `full`-Kovarianz kann bei größeren Zustandszahlen abbrechen
> (`covars must be symmetric, positive-definite`) — das ist das erwartete
> Instabilitäts-Verhalten aus Abschnitt 4, kein Fehler im Code.
