.. GestureRecognitionMPT documentation master file, created by
   sphinx-quickstart on Tue Mar 10 16:01:45 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GestureRecognitionMPT
=====================

Grundlagen der Gestenerkennung
==============================

Gestenerkennung ist ein Teilgebiet der Computer Vision und beschäftigt sich
mit der automatischen Interpretation menschlicher Bewegungen, insbesondere
von Händen und Körpern, durch algorithmische Verfahren. [1]_

Das Ziel ist es, aus sensorischen Daten (z. B. Kamerabilder oder Landmarken)
die zugrunde liegende Bedeutung einer Bewegung zu erkennen.


Statische vs. dynamische Gesten
-------------------------------

Ein zentrales Problem der Gestenerkennung ist die Unterscheidung zwischen
statischen und dynamischen Gesten.

- **Statische Gesten** bestehen aus einer einzelnen Pose (z. B. Handform)
- **Dynamische Gesten** bestehen aus einer zeitlichen Sequenz von Bewegungen

Dynamische Gesten sind deutlich komplexer, da sie nicht nur die Form,
sondern auch den zeitlichen Verlauf berücksichtigen müssen.

.. warning::

   Die meisten realen Gesten sind **dynamisch** und können nicht als
   einzelnes Bild korrekt erkannt werden.

Dies führt zu einer der wichtigsten Herausforderungen:

→ Gestenerkennung ist ein **Sequenzproblem**.


Typischer Aufbau eines Gestenerkennungssystems
----------------------------------------------

Ein modernes System besteht typischerweise aus mehreren Stufen:

1. **Detektion** (z. B. Handtracking)
2. **Feature-Extraktion** (z. B. Landmarken, Trajektorien)
3. **Sequenzmodellierung** (z. B. HMM, DTW, neuronale Netze)
4. **Klassifikation**

Diese Pipeline ist auch die Grundlage Ihrer Implementierung.


Herausforderungen der Gestenerkennung
-------------------------------------

Die wichtigsten Probleme sind:

- **Zeitliche Variation**
  - gleiche Geste → unterschiedliche Geschwindigkeit
- **Räumliche Variation**
  - unterschiedliche Position, Skalierung
- **Rauschen**
  - Trackingfehler, Sensorrauschen
- **Ähnliche Gesten**
  - gleiche Form, aber unterschiedliche Bewegung

Besonders dynamische Gesten enthalten viele Informationen
und sind daher deutlich schwieriger zu modellieren.


Historische Methoden
--------------------

Frühere Ansätze zur Gestenerkennung basierten häufig auf [2]_:

- Template Matching
- Regelbasierte Systeme
- Finite State Machines
- Dynamic Time Warping (DTW)

Diese Methoden funktionieren gut für einfache oder klar strukturierte Daten,
stoßen jedoch bei komplexen, variablen Sequenzen schnell an ihre Grenzen.


Moderne Ansätze
---------------

Heute werden hauptsächlich zwei Klassen von Methoden verwendet:

1. **Probabilistische Modelle**
   - Hidden Markov Models (HMM)
   - Gaussian Mixture Models

2. **Deep Learning**
   - Recurrent Neural Networks (RNN, LSTM)
   - Transformer-basierte Modelle

HMMs sind dabei besonders interessant, da sie:

- effizient trainierbar sind
- gut mit kleinen Datensätzen funktionieren
- zeitliche Struktur explizit modellieren
- einfach zu verstehen sind


Warum Daten entscheidend sind
-----------------------------

Ein Gestenerkennungssystem ist nur so gut wie die Daten, auf denen es basiert.

Im Gegensatz zu vielen klassischen Problemen hängt die Qualität der Ergebnisse
hier nicht primär vom gewählten Modell ab, sondern maßgeblich von der Qualität,
Struktur und Repräsentativität des Datensatzes.


Datenerhebung und Labeling
~~~~~~~~~~~~~~~~~~~~~~~~~~

Ein zentraler Schritt ist die systematische Erhebung und Annotation von Daten.

Dabei wird festgelegt:

- welche Gestenklassen existieren
- welche Aufnahmen zu welcher Klasse gehören
- wie konsistent diese Klassen voneinander abgegrenzt sind

Eine klare und konsistente Label-Struktur ist entscheidend, da das Modell
ausschließlich aus diesen Zuordnungen lernt.

Darüber hinaus ist es wichtig, eine effiziente Methodik zur Datenerhebung zu entwickeln:

- Je mehr Daten verfügbar sind, desto robuster wird das Modell
- Variationen in Geschwindigkeit, Ausführung und Position müssen abgedeckt werden
- Schlechte oder inkonsistente Aufnahmen wirken sich direkt negativ auf die Modellleistung aus

Ein effizienter Workflow zur Aufnahme und Organisation von Daten ermöglicht es,
schnell größere und qualitativ hochwertige Datensätze zu erstellen.


Visualisierung des Datensatzes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Die Visualisierung von Daten ist ein essenzieller Bestandteil des Entwicklungsprozesses.

Sie ermöglicht es, strukturelle Probleme im Datensatz frühzeitig zu erkennen:

- Ausreißer (fehlerhafte oder untypische Sequenzen)
- inkonsistente oder verrauschte Daten
- falsche Skalierung oder Normalisierung
- unerwartete Muster oder Artefakte

Insbesondere bei zeitlichen Daten wie Trajektorien ist es oft schwierig,
Probleme rein numerisch zu erkennen. Visualisierung macht diese direkt sichtbar.

Darüber hinaus hilft sie dabei:

- die Qualität der Feature-Extraktion zu beurteilen
- Unterschiede zwischen Klassen besser zu verstehen
- Annahmen über die Daten zu überprüfen


Evaluation und Iteration
~~~~~~~~~~~~~~~~~~~~~~~~

Die Entwicklung eines Gestenerkennungssystems ist ein iterativer Prozess.

Die Evaluation dient dazu:

- die Modellleistung objektiv zu messen
- Schwächen im Datensatz oder Modell zu identifizieren
- gezielt Verbesserungen vorzunehmen

Wichtige Aspekte sind:

- Vergleich zwischen Klassen
- Analyse von Fehlklassifikationen
- Verständnis, warum bestimmte Gesten verwechselt werden

Die Kombination aus:

- guter Datenerhebung
- systematischer Visualisierung
- kontinuierlicher Evaluation

führt in der Regel zu deutlich besseren Ergebnissen als die alleinige
Optimierung des Modells.


.. tip::

   In der Praxis ist die Verbesserung der Datenqualität häufig effektiver
   als die Verwendung komplexerer Modelle.

Hidden Markov Models (HMM)
--------------------------

Hidden Markov Models sind probabilistische Modelle zur Beschreibung von
Sequenzen.

Grundidee:

- Ein System befindet sich in einem von mehreren **Zuständen**
- Zustände sind **nicht direkt beobachtbar** (hidden)
- Jeder Zustand erzeugt Beobachtungen mit bestimmter Wahrscheinlichkeit

Ein HMM besteht aus:

- Zuständen (z. B. Phasen einer Geste)
- Übergangswahrscheinlichkeiten (State → State)
- Emissionswahrscheinlichkeiten (State → Beobachtung)


Funktionsweise (vereinfacht)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Eine Eingabesequenz (z. B. Trajektorie) wird beobachtet
2. Für jede Klasse existiert ein eigenes HMM
3. Für jedes Modell wird berechnet:

   → Wie wahrscheinlich ist es, dass dieses Modell die Sequenz erzeugt hat?

4. Die Klasse mit der höchsten Wahrscheinlichkeit wird gewählt


Warum funktionieren HMMs gut für Gesten?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gesten sind zeitliche Prozesse — ähnlich wie Sprache.

HMMs sind genau für solche Probleme entwickelt worden:

- sie modellieren Sequenzen explizit
- sie sind robust gegenüber Zeitverzerrungen
- sie können mit variabler Länge umgehen

HMMs wurden ursprünglich stark in der Spracherkennung eingesetzt,
da Sprachsignale ebenfalls als zeitliche, stochastische Prozesse
modelliert werden können.

Diese Eigenschaften machen sie ideal für:

- Handbewegungen
- Trajektorien
- zeitliche Muster


Zusammenfassung
---------------

- Gestenerkennung ist ein **Sequenzproblem**
- Dynamische Gesten sind deutlich komplexer als statische
- Datenqualität ist entscheidend für den Erfolg
- HMMs sind ein klassischer und effektiver Ansatz zur Modellierung von Gesten

Dieses theoretische Fundament ist entscheidend, um die praktische
Implementierung im weiteren Verlauf zu verstehen.

Das MPT-Projekt
===============

.. toctree::
   :maxdepth: 2

   pruefung

Literaturverzeichnis
====================
.. [1] https://de.wikipedia.org/wiki/Gestenerkennung
.. [2] Celebi, S. (2013).
       Gesture Recognition using Skeleton Data with Weighted Dynamic Time Warping.
       Proceedings of the International Conference on Computer Vision Theory and Applications.