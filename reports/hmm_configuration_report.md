# HMM Modellvergleich

## Warum Cross-Validation?

Cross-Validation ist robuster als ein einzelner Train/Test-Split,
weil jede Aufnahme einmal als Testdaten benutzt wird.
Dadurch haengt das Ergebnis weniger vom Zufall einer einzigen Aufteilung ab.

## Trade-offs bei n_components

- Wenige Zustaende: einfacher, stabiler, aber oft grober.
- Viele Zustaende: flexibler, aber leichteres Overfitting.
- Deshalb wird hier systematisch ueber mehrere Werte verglichen.

## Ergebnisse

| n_components | covariance_type | mean_accuracy | std_accuracy |
| --- | --- | --- | --- |
| 2 | diag | 1.0000 | 0.0000 |
| 2 | full | 1.0000 | 0.0000 |
| 3 | diag | 1.0000 | 0.0000 |
| 3 | full | 1.0000 | 0.0000 |

## Finale Wahl

- Gewaehlt wurde `n_components=2` mit `covariance_type=diag`.
- Die mittlere Accuracy liegt bei `1.0000`.
- Die Wahl ist einfach begruendet: beste mittlere Accuracy, bei gleicher Score-Hoehe waere die kleinere Standardabweichung besser.
