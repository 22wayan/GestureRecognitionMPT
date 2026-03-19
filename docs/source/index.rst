.. GestureRecognitionMPT documentation master file, created by
   sphinx-quickstart on Tue Mar 10 16:01:45 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GestureRecognitionMPT
=====================

Prüfungsleistung
================

In dieser Prüfung entwickeln Sie ein vollständiges System zur
Erkennung von Handgesten auf Basis von Hidden Markov Models (HMM).

Das Ziel ist es, eine durchgängige Pipeline zu implementieren:

- Datenerfassung
- Vorverarbeitung
- Modelltraining
- Evaluation
- Live-Inferenz

Die einzelnen Komponenten bauen aufeinander auf und müssen sauber
aufeinander abgestimmt sein.

.. warning::

   Diese Prüfung ist **systemorientiert**.
   Einzelne funktionierende Komponenten reichen nicht aus –
   entscheidend ist das Zusammenspiel aller Teile.


Aufgabenübersicht
-----------------

0. HandDetector & Preprocessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implementieren Sie Module zur:

- Erkennung von Händen und Landmarken
- Extraktion und Normalisierung von Fingertrajektorien

.. note::

   Diese Komponenten bilden die Grundlage für das gesamte System.

.. warning::

   Ihr Preprocessing muss so gestaltet sein, dass es später vom
   HMM-Classifier sinnvoll verarbeitet werden kann. An diesem Schritt
   sollten am Anfang alle Teilnehmer sinnvoll beteiligt sein.


1. Datenerfassung (Labeling)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Erstellen Sie ein System zur Aufnahme von Trainingsdaten.

Anforderungen:

- Neue Gesten sollen aufgezeichnet werden können
- Daten müssen strukturiert gespeichert werden
- Mehrere Klassen (Labels) müssen unterstützt werden

.. note::

   In der Prüfung werden neue Daten **live vom Prüfer aufgenommen**.
   Ihr System muss darauf vorbereitet sein.

.. tip::

   Denken Sie über einen effizienten Workflow nach:
   - schnelle Aufnahme
   - einfaches Verwerfen schlechter Sequenzen
   - klare Datenorganisation


2. Datenexploration & Visualisierung
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sie müssen Ihren Datensatz analysieren und verstehen können.

Beispiele:

- Visualisierung von Trajektorien (z. B. als Plot)
- Vergleich mehrerer Sequenzen pro Klasse

Zusätzlich:

- Darstellung der Modellperformance (z. B. Confusion Matrix)

.. note::

   Gute Modelle entstehen nur mit guten Daten.

.. tip::

   Nutzen Sie Visualisierung aktiv zum Debugging.


3. HMMClassifier (Training & Inferenz)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implementieren Sie einen eigenen Klassifikator basierend auf
Hidden Markov Models.

Anforderungen:

- Trainieren Sie ein Modell pro Klasse
- Klassifizieren Sie Sequenzen anhand ihrer Wahrscheinlichkeit
- Wählen Sie die Klasse mit dem besten Score

.. warning::

   Der Klassifikator muss **selbst implementiert werden**.
   Es reicht nicht, fertige Lösungen zu verwenden.

.. tip::

   Überlegen Sie:
   - Wie strukturieren Sie Ihre Trainingsdaten?
   - Wie vergleichen Sie Modelle?
   - Wie gehen Sie mit Sequenzlängen um?

Erweiterung (optional):

- Grid Search für Hyperparameter
  (z. B. Anzahl Zustände, Modellstruktur)
- Vergleich verschiedener Modellkonfigurationen


4. Live-Modus
~~~~~~~~~~~~~

Ihr System soll in der Lage sein:

- Live-Daten aufzunehmen
- Diese direkt zu verarbeiten
- Eine Geste in Echtzeit zu klassifizieren

.. note::

   Dies ist der finale Integrationstest Ihres Systems.

.. warning::

   Alle Komponenten müssen hier zuverlässig zusammenspielen:
   - Detector
   - Preprocessor
   - Classifier


Bewertungskriterien
-------------------

Die Bewertung orientiert sich an folgenden Punkten:

- Funktionalität des Gesamtsystems
- Qualität und Struktur der Daten
- Verständlichkeit und Nachvollziehbarkeit
- Robustheit der Lösung
- Qualität der Modellperformance

.. tip::

   Eine sehr gute Lösung zeichnet sich dadurch aus, dass:
   - das System stabil läuft
   - die Daten sauber aufbereitet sind
   - die Ergebnisse nachvollziehbar erklärt werden können

.. note::

   Bonuspunkte können durch weiterführende Ansätze erzielt werden,
   wie z. B. Hyperparameter-Optimierung oder zusätzliche Analysen.


Dokumentation
-------------

Sie müssen in der Lage sein, Ihr System zu erklären:

- Aufbau der Pipeline
- Entscheidungen im Design
- Interpretation der Ergebnisse

.. warning::

   Unklare oder nicht nachvollziehbare Lösungen führen zu Punktabzug.

.. toctree::
   :maxdepth: 2
   :caption: Implemtierung:

   modules
   labeling
   visualization
   hmmclassifier

