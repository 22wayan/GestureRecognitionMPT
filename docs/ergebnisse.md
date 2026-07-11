# Ergebnisse: Wie gut erkennt das Modell die Gesten?

Diese Seite fasst zusammen, **wie gut unser trainiertes Modell die
Buchstaben-Gesten (A–Z) erkennt**. Sie richtet sich bewusst an Einsteiger:
jeder Fachbegriff wird kurz erklärt.

Alle Zahlen stammen aus der Funktion
[`evaluate_classifier()`](../GestureRecognition/visualization.py) und wurden auf
unserem eigenen Datensatz gemessen (26 Klassen A–Z, Aufnahmen von mehreren
Personen).

---

## 1. Was messen wir überhaupt? (Accuracy)

**Accuracy** (deutsch: Genauigkeit) ist einfach der Anteil der richtig
erkannten Gesten:

```
Accuracy = richtig erkannte Gesten / alle getesteten Gesten
```

Beispiel: Wenn das Modell von 100 Gesten 79 richtig errät, ist die
Accuracy = 79 %.

Wir berechnen **zwei** verschiedene Accuracy-Werte, weil sie zwei
unterschiedliche Fragen beantworten. Das ist der wichtigste Punkt dieser Seite.

### a) Standard-Accuracy — „kenne ich diese Person schon?"

Hier trainieren und testen wir mit **denselben Personen**, nur mit
unterschiedlichen Aufnahmen. Das Modell hat also von jeder Person schon Beispiele
gesehen.

> **Ergebnis: ~79 % richtig** (getestet auf 210 Test-Gesten, trainiert auf 839).

Das ist die *optimistische* Zahl.

### b) Neue-Person-Accuracy — „was, wenn eine völlig fremde Person kommt?"

Hier halten wir **eine ganze Person komplett aus dem Training heraus**
(bei uns: „yannik") und testen das Modell nur an dieser Person. Das Modell hat
diese Hand also **noch nie gesehen** – genau wie in der Prüfung, wenn der Prüfer
live neue Gesten aufnimmt.

> **Ergebnis: ~36 % richtig** (getestet auf 385 Gesten dieser einen Person).

Das ist die *ehrliche* Zahl für die Prüfungssituation.

### Warum ist die zweite Zahl so viel schlechter?

Weil **jeder Mensch einen Buchstaben etwas anders in die Luft malt** – andere
Größe, andere Geschwindigkeit, andere Richtung. Solange das Modell eine Person
schon kennt, ist das kein Problem. Bei einer völlig neuen Person muss es aber von
den bekannten Personen auf die neue *verallgemeinern* – und das fällt schwer.

**Wichtigste Erkenntnis:** Mehr **verschiedene Personen** im Training würden die
Neue-Person-Genauigkeit am stärksten verbessern – deutlich mehr als ein
komplizierteres Modell.

---

## 2. Die Confusion Matrix (Verwechslungs-Tabelle)

Eine **Confusion Matrix** zeigt nicht nur *ob*, sondern *welche* Buchstaben das
Modell verwechselt.

![Confusion Matrix des Standard-Tests](source/_static/confusion_matrix.png)

**So liest man die Tabelle:**

- Jede **Zeile** = der Buchstabe, der wirklich gemacht wurde (*True label*).
- Jede **Spalte** = der Buchstabe, den das Modell geraten hat (*Predicted label*).
- Die **Diagonale von links oben nach rechts unten** = alles richtig erkannt.
  Je dunkler die Diagonale, desto besser.
- Jede Zahl **neben** der Diagonale ist eine **Verwechslung**.

Beispiel: In der Zeile „P" steht eine „3" in der Spalte „F". Das heißt:
**3-mal wurde ein „P" gemacht, aber das Modell hat „F" erkannt.**

---

## 3. Was verwechselt das Modell am häufigsten?

Aus der Matrix abgelesen (wahr → vom Modell geraten):

| Verwechslung | Anzahl | Mögliche Erklärung |
|---|---|---|
| P → F | 3× | Ähnlicher Bewegungsanfang (senkrechter Strich) |
| N → P | 3× | Ähnliche Auf-/Ab-Bewegung |
| E → S | 3× | Beide bestehen aus kurzen Kurven |
| S → A | 2× | Runde Bewegung wird verwechselt |
| R → K | 2× | Ähnliche Ecken/Richtungswechsel |
| K → X | 2× | Beide haben kreuzende Bewegungen |
| G → S | 2× | Runde Form |

Das sind fast immer **Buchstaben mit ähnlichem Bewegungsverlauf** – ein Zeichen,
dass das Modell sinnvoll (nach der Form der Bewegung) unterscheidet und nicht
zufällig rät.

---

## 4. Welche Buchstaben laufen gut, welche schlecht?

**Sehr gut erkannt (100 %):** F, I, Q, U, V, W, Z
**Gut (~85–90 %):** A, D, O, T, Y
**Schwach:** N (38 %), R (38 %), E (50 %), G / K / P (je 62 %)

Die schwachen Buchstaben sind genau die aus der Verwechslungs-Tabelle oben. Wer
das Modell verbessern will, sollte **für diese Buchstaben mehr und sauberere
Aufnahmen** sammeln.

---

## 5. Ergebnisse selbst reproduzieren

Die Zahlen und das Bild lassen sich jederzeit neu erzeugen. Im Projekt-`.venv`
(dort ist `hmmlearn` installiert):

```python
# evaluate_classifier trainiert das Modell frisch, testet es und
# speichert die Confusion-Matrix als PNG-Bild ab.
from GestureRecognition.visualization import evaluate_classifier

ergebnis = evaluate_classifier(
    output_dir="docs/source/_static",  # hier wird confusion_matrix.png gespeichert
    held_out_person="yannik",          # diese Person wird fuer den 2. Test rausgehalten
)
print(ergebnis)
# -> {'accuracy_standard': 0.79..., 'accuracy_new_person': 0.36..., 'held_out_person': 'yannik'}
```

> **Hinweis:** Das Training eines HMM enthält kleine Zufallsanteile. Die Werte
> können sich daher von Lauf zu Lauf um ein paar Prozentpunkte unterscheiden – die
> hier gezeigten Zahlen sind das Ergebnis eines repräsentativen Laufs.

---

## 6. Kurzfazit

- Bekannte Personen werden zuverlässig erkannt (**~79 %**).
- Eine völlig neue Person ist deutlich schwerer (**~36 %**) – das ist die ehrliche
  Prüfungs-Erwartung.
- Verwechselt werden vor allem Buchstaben mit **ähnlicher Bewegung** – das Modell
  arbeitet also plausibel.
- Größter Hebel für Verbesserung: **mehr verschiedene Personen** im Training.
