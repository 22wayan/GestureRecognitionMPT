# Alle SignalHub-Module des Projekts an einer Stelle exportieren,
# damit demo.py sie einfach per "from GestureRecognition.modules import *" laedt.
from .trailmarker import TrailMarker
from .hiddenmarkov import HMMModule
from .handdetector import HandDetector
from .preprocessor import Preprocessor