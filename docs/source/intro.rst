Erste Schritte
==============

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
~~~~~~~~~~~~~~~~~~~~~~~~

Um effizient mit der Aufgabe arbeiten zu können, ist es wichtig,
die grundlegenden Konzepte des bereitgestellten Frameworks zu verstehen.

.. note::

    Sie müssen das Framework **nicht vollständig verstehen**, um zu starten.
    Wichtiger ist:

    - Welche Daten bekommt mein Modul?
    - Welche Daten muss ich zurückgeben?

.. toctree::

    signalhub