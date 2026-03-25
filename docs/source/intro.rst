Erste Schritte
--------------

Bevor Sie mit der Implementierung beginnen, müssen Sie das Projekt lokal einrichten.

1. Repository klonen:

.. code-block:: bash

   git clone https://github.com/jaboll-ai/GestureRecognitionMPT
   cd GestureRecognitionMPT

2. Abhängigkeiten installieren:

.. code-block:: bash

   pip install -r requirements.txt

.. tip::

    Falls Sie ``uv`` verwenden, können Sie die Umgebung auch automatisch verwalten

    .. code-block:: bash

        uv sync

3. Download der Recording-Dateien

Die bereitgestellten Daten finden Sie
`hier <https://github.com/jaboll-ai/GestureRecognitionMPT/releases/tag/recordings-v1>`_.

Diese können Sie entweder im Browser oder über die Kommandozeile herunterladen:

.. code-block:: bash

   wget https://github.com/jaboll-ai/GestureRecognitionMPT/releases/download/recordings-v1/recordings.zip

Entpacken Sie die ``.7z``- oder ``.zip``-Datei in Ihren geklonten Projektordner.

4. Testlauf im Replay-Modus:

.. code-block:: bash

   python main.py --mode replay --recorder <path_to_recording>

Ersetzen Sie ``<path_to_recording>`` durch eine der bereitgestellten
Recording-Dateien.

.. note::

   Der Replay-Modus ist der einfachste Einstiegspunkt, da er keine Webcam benötigt.


Grundlagen zum Framework
------------------------

Um effizient mit der Aufgabe arbeiten zu können, ist es wichtig,
die grundlegenden Konzepte des bereitgestellten Frameworks zu verstehen.


SignalHub
~~~~~~~~~

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


Replay-Modus
~~~~~~~~~~~~

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^

Das bedeutet konkret:

- Module im ``replay``-Block erhalten ihre Daten aus der Aufnahme
- Alle anderen Module erzeugen ihre Daten wie gewohnt live

Dies erlaubt es, gezielt einzelne Teile der Pipeline zu testen.

Beispiel:

- Sie verwenden aufgezeichnete ``detector``-Daten
- und entwickeln darauf Ihren ``preprocessor`` unabhängig weiter


Datenfluss im Replay-Modus
^^^^^^^^^^^^^^^^^^^^^^^^^^

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GALY steht für *Graphical Abstraction Layer*.

Es handelt sich um ein einfaches Interface zur Visualisierung von Daten
innerhalb der Pipeline.

Mit GALY können Sie:

- Punkte (z. B. Fingerpositionen) zeichnen
- Linien (Trajektorien) darstellen
- Text (z. B. Klassifikationsergebnisse) anzeigen

GALY wird typischerweise verwendet für:

- Debugging von Trajektorien
- Darstellung von Zwischenschritten
- Visualisierung der Klassifikation


Wie Sie starten sollten
~~~~~~~~~~~~~~~~~~~~~~~

Sie müssen das Framework **nicht vollständig verstehen**, um zu starten.

Wichtiger ist:

- Welche Daten bekommt mein Modul?
- Welche Daten muss ich zurückgeben?

Der Rest ergibt sich iterativ während der Entwicklung.

.. tip::

   Nutzen Sie den Replay-Modus intensiv.
   Er ist der wichtigste Baustein, um unabhängig und effizient
   an Ihrer Lösung zu arbeiten.