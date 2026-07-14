"""
Einstiegspunkt der Anwendung: startet die SignalHub-Pipeline mit unseren Modulen.

Benutzung
---------
    python main.py --mode live      # Webcam an, Gesten werden live erkannt
    python main.py --mode record    # Webcam an, Aufnahme wird gespeichert
    python main.py                  # Standard: Live-Betrieb mit Webcam

Welche Module laufen und in welcher Reihenfolge, steht in
``GestureRecognition/__init__.py`` und ``config.yml``.
"""

import argparse

from GestureRecognition import run

parser = argparse.ArgumentParser("GestureRecognition")

def main():
    run(parser)

if __name__ == "__main__":
    main()
