SignalHub
=========

SignalHub ist ein modulares Pipeline-System.

- Daten werden in Form von *Signalen* zwischen Modulen weitergegeben
- Jedes Modul verarbeitet eingehende Daten und erzeugt neue Signale
- Die Pipeline wird über die ``config.yaml`` definiert

Typische Signale für dieses Projekt sind z. B.:

- ``webcam`` → Rohbild der Kamera
- ``detector`` → erkannte Hand + Landmarken
- ``preprocessor`` → normalisierte Trajektorie

Ziel ist es, Module zu implementieren, die sauber miteinander
über diese Signale interagieren.

Signalbasierte Kommunikation
----------------------------

Module kommunizieren ausschließlich über sogenannte *Signale*.

Dieses Konzept ist **zentral für das gesamte Framework** und muss
unbedingt verstanden werden.


Grundprinzip
~~~~~~~~~~~~

Ein Modul definiert explizit:

- welche Signale es **empfängt** (``inputSignals``)
- welche Signale es **erzeugt** (``outputSchema``)

Beispiel:

.. code-block:: python

   def step(self, data):
       detector_data = data["detector"]
       return {"preprocessor": result}


Wichtige Regeln
~~~~~~~~~~~~~~~

**1. Sie erhalten nur die Signale, die Sie abonnieren**

- Das ``data``-Dictionary enthält **ausschließlich** die Signale,
  die in ``inputSignals`` definiert wurden
- Ein Zugriff auf nicht abonnierte Signale ist **nicht möglich**

.. warning::

   Wenn Sie ein Signal nicht in ``inputSignals`` angeben,
   existiert es für Ihr Modul nicht.


**2. Sie dürfen nur definierte Outputs zurückgeben**

- Die Rückgabe Ihres Moduls muss zum ``outputSchema`` passen
- Nicht definierte Outputs werden ignoriert oder führen zu Fehlern

.. warning::

   Sie können **keine beliebigen Keys** zurückgeben.
   Jeder Output muss vorher im Schema definiert sein.


**3. Datenfluss ist strikt gerichtet**

- Module arbeiten **nicht global**
- Es gibt keinen direkten Zugriff auf andere Module
- Kommunikation erfolgt ausschließlich über Signale


Warum ist das wichtig?
~~~~~~~~~~~~~~~~~~~~~~

Dieses Konzept erzwingt eine klare Struktur:

- Module sind voneinander entkoppelt
- Datenflüsse sind explizit sichtbar
- Fehler sind leichter nachvollziehbar

Gleichzeitig bedeutet das:

- falsche Signalnamen → kein Zugriff
- fehlende Subscriptions → leere Daten
- falsche Outputs → Pipeline bricht


Replay-Modus
------------

Neben Live-Daten können auch aufgezeichnete Daten verwendet werden.

Dafür existiert ein sogenannter *Replay-Modus*:

- Anstelle der Webcam werden gespeicherte Aufnahmen abgespielt
- Dies ermöglicht reproduzierbares Testen und Debugging

Der Replay-Modus kann über die Kommandozeile gestartet werden:

.. code-block:: bash

   python main.py --mode replay --recorder <path_to_recording>

Dies ist besonders wichtig, um:

- den Preprocessor unabhängig zu entwickeln
- Visualisierungen zu testen
- den Classifier ohne Live-Input zu debuggen


Steuerung über die Konfiguration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Welche Daten tatsächlich *replayed* werden, wird über die ``config.yaml`` gesteuert.

Beispiel:

.. code-block:: yaml

   recorder:
     replay:
       - trailmarker
       - preprocessor
       - hiddenmarkov
       - detector

Nur die hier aufgeführten Module bzw. Signale werden aus den
aufgezeichneten Daten gespeist.

Alle anderen Module laufen weiterhin im **Live-Modus**.


Live- vs. Replay-Daten
~~~~~~~~~~~~~~~~~~~~~~

Das bedeutet konkret:

- Module im ``replay``-Block erhalten ihre Daten aus der Aufnahme
- Alle anderen Module erzeugen ihre Daten wie gewohnt live

Dies erlaubt es, gezielt einzelne Teile der Pipeline zu testen.

Beispiel:

- Sie verwenden aufgezeichnete ``detector``-Daten
- und entwickeln darauf Ihren ``preprocessor`` unabhängig weiter


Datenfluss im Replay-Modus
~~~~~~~~~~~~~~~~~~~~~~~~~~

Die aufgezeichneten Daten werden **schrittweise (Frame für Frame)** in die
Pipeline eingespeist.

Das bedeutet:

- Ihr Modul erhält pro ``step`` genau die Daten, die zum jeweiligen Zeitpunkt
  aufgezeichnet wurden
- Der zeitliche Verlauf der Sequenz bleibt erhalten

.. note::

   Der Replay-Modus ersetzt nicht die gesamte Pipeline, sondern nur die
   explizit konfigurierten Teile. Dies ermöglicht eine sehr flexible
   Kombination aus Live- und aufgezeichneten Daten.


Recordings (bereitgestellter Datensatz)
---------------------------------------

Die bereitgestellten Recordings stammen aus SignalHub.

- Sie enthalten Rohdaten aus der Pipeline (z. B. Detector-Ausgaben)
- Sie bilden die Grundlage für Ihre Entwicklung

.. warning::

    Die bereitgestellten Daten dienen lediglich als Grundlage, damit Sie
    parallel an unterschiedlichen Schritten der Pipeline arbeiten können.

    Ziel des Projekts ist es ausdrücklich, eigene Daten aufzuzeichnen,
    auf deren Basis Modelle zu trainieren und diese anschließend auch
    zu evaluieren.


GALY (Graphical Abstraction Layer)
----------------------------------

GALY steht für *Graphical Abstraction Layer*.

Im Kern ist GALY ein Wrapper um die wichtigsten Zeichenfunktionen aus ``cv2`` (OpenCV).
Der entscheidende Unterschied ist jedoch:

**Zeichenoperationen werden nicht direkt ausgeführt, sondern als Befehle gespeichert.**

Diese Befehle können:

- serialisiert (gespeichert)
- deserialisiert (wieder abgespielt)

werden.

Dadurch ist es möglich, **Visualisierungen zusammen mit den Daten aufzuzeichnen**
und später im Replay-Modus exakt wiederzugeben.


Grundprinzip
~~~~~~~~~~~~

Anstatt direkt auf ein Bild zu zeichnen, werden Zeichenbefehle gesammelt:

.. code-block:: python

   galy = GALY()
   galy.line((0, 0), (100, 100), (255, 0, 0))

Intern wird dabei kein ``cv2.line`` direkt ausgeführt, sondern ein Befehl gespeichert.

Diese Befehle werden später vom Framework verarbeitet und gerendert.


Canvas und Layer
~~~~~~~~~~~~~~~~

GALY arbeitet mit zwei zentralen Konzepten:

**Canvas**
   Eine Zeichenfläche (vergleichbar mit einem Bild)

**Layer**
   Eine Ebene auf dem Canvas (vergleichbar mit Photoshop-Layern)

Beispiel:

.. code-block:: python

    galy = GALY()
    galy.canvas("main", (640, 480), (0, 0, 0))
    galy.layer("trajectory")

- ``canvas`` erstellt eine Zeichenfläche
- ``layer`` erlaubt strukturierte Visualisierung

Layer können ein- und ausgeblendet werden und erleichtern Debugging.


Grundlegende Zeichenoperationen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Linie zeichnen:

.. code-block:: python

   galy.line((x1, y1), (x2, y2), (255, 0, 0), thickness=2)

Kreis zeichnen:

.. code-block:: python

   galy.circle((x, y), radius=5, color=(0, 255, 0), thickness=2)

Text anzeigen:

.. code-block:: python

   galy.putText("Label", (x, y), color=(255, 255, 255))

Diese Funktionen entsprechen direkt den OpenCV-Funktionen,
werden jedoch über GALY abstrahiert.


Bilder einfügen (Blitting)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Mit ``blit`` können komplette Bilder auf den Canvas gelegt werden:

.. code-block:: python

   galy.blit("webcam", (0, 0))

Dabei wird das Bild aus einem Signal (z. B. ``webcam``) verwendet.

Dies ist typischerweise der erste Schritt, um das Kamerabild darzustellen,
auf das anschließend weitere Elemente gezeichnet werden.


Affine Transformationen
~~~~~~~~~~~~~~~~~~~~~~~

GALY unterstützt affine Transformationen pro Layer.

.. code-block:: python

   mapping = np.array([
       [1.0, 0.0, tx],
       [0.0, 1.0, ty]
   ])

   galy.set_layer_affine_mapping(mapping)

Eine affine Transformation ist eine lineare Transformation der Form:

- Translation (Verschiebung)
- Skalierung
- Rotation

Diese wird auf alle Punkte eines Layers angewendet.

Beispiel:

- Koordinaten normalisiert → Mapping auf Bildschirm
- Verschiebung der gesamten Trajektorie
- Skalierung von Daten

Intern wird jeder Punkt transformiert, bevor er gezeichnet wird.


Mahalanobis-Visualisierung
~~~~~~~~~~~~~~~~~~~~~~~~~~

GALY enthält auch spezialisierte Visualisierung:

.. code-block:: python

   galy.mahalanobis(mean, covariance, color=(255, 0, 0))

Dies zeichnet eine Ellipse basierend auf:

- Mittelwert (mean)
- Kovarianzmatrix (covariance)

Dies kann z. B. verwendet werden für:

- Visualisierung von Zuständen im HMM
- Darstellung von Unsicherheiten

Integration in Module
~~~~~~~~~~~~~~~~~~~~~

Damit GALY-Visualisierungen im Framework angezeigt werden,
muss das erzeugte GALY-Objekt vom Modul zurückgegeben werden.

Beispiel:

.. code-block:: python

   def step(self, data):
       galy = GALY()

       galy.canvas("main", (640, 480), (0, 0, 0))
       galy.layer("debug")

       galy.line((0, 0), (100, 100), (255, 0, 0))

       return {
           ...
           "galy": galy
       }

Wichtig:

- Das Signal **muss genau** ``"galy"`` **heißen**
- Nur dann wird es vom Framework automatisch verarbeitet

.. note::

   Wenn kein ``galy``-Objekt zurückgegeben wird,
   erfolgt auch keine Visualisierung.

.. tip::

   Sie können GALY in jedem Modul verwenden.
   Besonders hilfreich ist dies im Preprocessor und Classifier,
   um Zwischenergebnisse sichtbar zu machen.


Verarbeitung im Framework
~~~~~~~~~~~~~~~~~~~~~~~~~

Alle gesammelten Befehle werden später verarbeitet:

.. code-block:: python

   process_galy(data)

Dabei passiert:

1. Alle Commands werden iteriert
2. In echte ``cv2``-Aufrufe übersetzt
3. Auf Canvas gerendert
4. In GUI angezeigt

Dadurch bleibt die Visualisierung:

- reproduzierbar
- speicherbar
- unabhängig vom Zeitpunkt der Berechnung


Wichtige Hinweise
~~~~~~~~~~~~~~~~~

- GALY ersetzt **nicht** OpenCV, sondern abstrahiert es
- Alle Zeichenoperationen sind **zustandslos gesammelt**
- Reihenfolge der Befehle ist entscheidend
- Visualisierung kann Teil der aufgezeichneten Daten sein

.. tip::

   Nutzen Sie GALY aktiv zum Debugging.
   Eine gute Visualisierung hilft oft mehr als jede Konsole.


Pipeline verändern
------------------

SignalHub organisiert die gesamte Verarbeitung als Pipeline aus Modulen.

Die Pipeline wird explizit im Code definiert, z. B.:

.. code-block:: python

   modules = [
       ConfigParser(parser),
       Webcam(),
       HandDetector(),
       TrailMarker(),
       Preprocessor(),
       HMMModule(),
   ]

   engine = Engine(modules=modules, signals={})
   signals = engine.run({})


Pipeline-Struktur
-----------------

Durch Ändern dieser Liste können Sie die gesamte Pipeline anpassen.

Beispiele:

- Module hinzufügen (z. B. eigene Visualisierung)
- Module entfernen (z. B. TrailMarker deaktivieren)
- Module austauschen (z. B. eigener Classifier)

.. note::

   Die Pipeline ist vollständig flexibel und kann beliebig erweitert oder verändert werden.


Eigene Module erstellen
-----------------------

Neue Module werden erstellt, indem Sie von der Basisklasse ``Module`` erben:

.. code-block:: python

   from SignalHub import Module

   class MyModule(Module):

       def start(self, data):
           return {}

       def step(self, data):
           return {}

       def stop(self, data):
           pass


Jedes Modul besteht aus drei zentralen Methoden:

- ``start`` → wird einmal beim Start aufgerufen
- ``step`` → wird für jeden Frame ausgeführt
- ``stop`` → wird beim Beenden aufgerufen

Erweiterung der Pipeline
------------------------

Sie können jederzeit eigene Module in die Pipeline integrieren:

.. code-block:: python

   modules = [
       ConfigParser(parser),
       Webcam(),
       HandDetector(),
       ##########
       MyModule(),
       ##########
       Preprocessor(),
       HMMModule(),
   ]

Wichtig ist dabei:

- das Modul korrekt Signale abonniert
- die erwarteten Datenstrukturen einhält


.. tip::

   Beginnen Sie mit kleinen Modulen und erweitern Sie die Pipeline schrittweise.
   So lassen sich Fehler deutlich einfacher finden.


Zusammenfassung
---------------

- Die Pipeline wird über eine Liste von Modulen definiert
- Module werden in Reihenfolge ausgeführt
- Neue Module entstehen durch Vererbung von ``Module``
- Kommunikation erfolgt über Signale und das ``data``-Dictionary

Dieses Konzept erlaubt eine sehr flexible und erweiterbare Architektur.

