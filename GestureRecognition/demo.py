from SignalHub import Engine, ConfigParser, Webcam
from GestureRecognition.modules import *
import argparse

def run(parser: argparse.ArgumentParser):
    parser.add_argument("--mode", action="store", default="none")
    parser.add_argument("--recorder.file", action="store")
    parser.add_argument("--engine.singlestep", action="store_true", default=False)
    parser.add_argument("--webcam.width", required=False)

    # Mode vorab auslesen, um die Modulliste passend zusammenzustellen.
    args, _ = parser.parse_known_args()
    mode = args.mode or ""

    modules = [ConfigParser(parser)]
    # Im Replay-Modus liefern die Replay-Wrapper die Daten aus einer Aufnahme,
    # eine Webcam ist dann nicht noetig (und wuerde unnoetig die Kamera oeffnen).
    # Fuer Record- und Live-Betrieb wird die Webcam als Bildquelle gebraucht.
    if "replay" not in mode:
        modules.append(Webcam())
    modules += [
        HandDetector(),
        TrailMarker(),
        Preprocessor(),
        HMMModule(),
    ]
    engine = Engine(modules=modules, signals={})
    signals = engine.run({})

if __name__ == "__main__":
    parser = argparse.ArgumentParser("GestureRecognition")
    run(parser)