import os


def _fix_qt_plugin_path() -> None:
    """Sorgt dafuer, dass Qt seine Plattform-Plugins auch dann findet, wenn der
    Projektpfad Nicht-ASCII-Zeichen enthaelt.

    Hintergrund
    -----------
    SignalHub oeffnet sein Anzeigefenster ueber PyQt5. Qt ermittelt den Pfad zu
    seinen Plattform-Plugins (u. a. ``cocoa`` auf macOS) selbst aus dem
    Installationsort. Enthaelt dieser Pfad Nicht-ASCII-Zeichen -- etwa das
    ``ue`` in einem OneDrive-Ordner wie "...HochschuleDuesseldorf..." --
    verstuemmelt Qt das Zeichen intern (``ue`` -> ``?``) und sucht die Plugins
    in einem Ordner, den es so gar nicht gibt. Folge: die Aufnahme bricht mit

        qt.qpa.plugin: Could not find the Qt platform plugin "cocoa" in ""

    sofort ab und es entsteht keine .pkl-Datei.

    Python kommt mit dem Pfad problemlos zurecht. Wir bestimmen den
    Plugin-Ordner darum hier und reichen ihn ueber die Umgebungsvariable
    ``QT_QPA_PLATFORM_PLUGIN_PATH`` an Qt weiter -- als UTF-8-Bytes, die Qt
    korrekt liest. Das muss geschehen, *bevor* eine ``QApplication`` erzeugt
    wird, also vor dem Import von SignalHub/demo weiter unten.
    """
    # Eine bereits gesetzte Variable (z. B. bewusst vom Benutzer) nicht ueberschreiben.
    if os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"):
        return

    try:
        import PyQt5
    except ImportError:
        # Ohne PyQt5 gibt es kein Qt-Fenster -- dann ist nichts zu reparieren.
        return

    plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
    if os.path.isdir(plugin_path):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path


_fix_qt_plugin_path()

from .demo import run
