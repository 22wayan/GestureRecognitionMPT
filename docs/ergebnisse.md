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

Beispiel: Wenn das Modell von 100 Gesten 90 richtig errät, ist die
Accuracy = 90 %.

Wir berechnen **zwei** verschiedene Accuracy-Werte, weil sie zwei
unterschiedliche Fragen beantworten. Das ist der wichtigste Punkt dieser Seite.

### a) Standard-Accuracy — „kenne ich diese Person schon?"

Hier trainieren und testen wir mit **denselben Personen**, nur mit
unterschiedlichen Aufnahmen. Das Modell hat also von jeder Person schon Beispiele
gesehen.

> **Ergebnis: ~90 % richtig** (getestet auf 350 Test-Gesten, trainiert auf 1399).

Das ist die *optimistische* Zahl.

### b) Neue-Person-Accuracy — „was, wenn eine völlig fremde Person kommt?"

Hier halten wir **eine ganze Person komplett aus dem Training heraus**
(bei uns: „yannik") und testen das Modell nur an dieser Person. Das Modell hat
diese Hand also **noch nie gesehen** – genau wie in der Prüfung, wenn der Prüfer
live neue Gesten aufnimmt.

> **Ergebnis: ~60 % richtig** (getestet auf 434 Gesten dieser einen Person).
> Je nachdem, welche Person man raushält, liegt der Wert zwischen ~58 % und ~77 %.

Das ist die *ehrliche* Zahl für die Prüfungssituation — **außer** die neue
Person nimmt vorher ein paar eigene Aufnahmen auf und trainiert mit
(`schnell_aufnahme.py` + `python train.py`). Dann gilt die Standard-Zahl.

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
| G → Q | 4× | Beide: runde Form mit „Schwänzchen" am Ende |
| Q → D | 3× | Runde Grundform, ähnlicher Verlauf |
| O → D | 2× | Beide fast geschlossene Rundung |
| Z → Q, Y → X, X → Q | je 1× | Einzelfälle |

Es bleiben fast nur noch **runde Buchstaben** übrig, deren Bewegungsspur sich
wirklich ähnelt (G/Q/O/D) – ein Zeichen, dass das Modell sinnvoll nach der
Form der Bewegung unterscheidet und nicht zufällig rät. Die früher häufigen
Verwechslungen (P→F, N→P, E→S) sind durch das Resampling und mehr
Trainingsdaten verschwunden.

---

## 4. Welche Buchstaben laufen gut, welche schlecht?

**Perfekt erkannt (100 %):** A, B, D, E, F, H, I, J, M, N, R, V, W
**Gut (85–95 %):** C, K, L, O, P, Q, S, T, U, X, Y, Z
**Schwächster Buchstabe:** G (67 %, wird mit Q verwechselt)

Wer das Modell weiter verbessern will, sollte also vor allem **für G (und die
runden Nachbarn Q, O, D) mehr und deutlich unterscheidbare Aufnahmen** sammeln
— z. B. das „Schwänzchen" des G bewusst betonen.

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
# -> {'accuracy_standard': 0.90..., 'accuracy_new_person': 0.60..., 'held_out_person': 'yannik'}
```

Die drei Demo-GIFs im README entstehen übrigens genauso reproduzierbar:

```bash
python demo_gif.py            # nutzt data/hmm.pkl, schreibt images/demo_*.gif
```

> **Hinweis:** Das Training eines HMM enthält kleine Zufallsanteile. Die Werte
> können sich daher von Lauf zu Lauf um ein paar Prozentpunkte unterscheiden – die
> hier gezeigten Zahlen sind das Ergebnis eines repräsentativen Laufs.

---

## 6. Kurzfazit

- Bekannte Personen werden zuverlässig erkannt (**~90 %**, 13 von 26
  Buchstaben sogar fehlerfrei).
- Eine völlig neue Person ist deutlich schwerer (**~60 %**) – das ist die
  ehrliche Erwartung, wenn jemand ohne eigene Trainingsdaten loslegt. Mit ein
  paar eigenen Aufnahmen + `python train.py` gilt die 90-%-Zahl.
- Verwechselt werden fast nur noch **runde Buchstaben** (G/Q/O/D) – das Modell
  arbeitet also plausibel.
- Größte Hebel waren: **Resampling auf feste Gestenlänge**, **min_covar**
  gegen den HMM-Kollaps und **mehr verschiedene Personen** im Training
  (siehe [design-entscheidungen.md](design-entscheidungen.md)).
