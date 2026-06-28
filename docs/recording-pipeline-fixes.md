# Visualisierung & Aufnahme-Pruefung (Fixes + Tool)

Dieses Dokument erklaert kurz, **was** geaendert wurde und **warum**. Es ging
darum, das Aufnehmen der Buchstaben tatsaechlich benutzbar zu machen.

## Hintergrund: normalisierte Punkte vs. Pixel

MediaPipe gibt jeden Handpunkt als Zahl zwischen `0` und `1` zurueck (relativ
zur Bildgroesse). GALY zeichnet aber in **echten Pixeln**. Wenn man die `0..1`-
Werte direkt zeichnet, landet alles in der Ecke bei Pixel `(0, 0)` und ist
unsichtbar. Loesung: dem Layer eine kleine Umrechnung mitgeben
(`set_layer_affine_mapping`), die `x` mit der Bildbreite und `y` mit der
Bildhoehe multipliziert.

## Behobene Fehler

1. **Hand-Skelett unsichtbar** – `HandDetector` ([modules/handdetector.py](../GestureRecognition/modules/handdetector.py))
   setzt jetzt das Pixel-Mapping fuer den `hands`-Layer.
2. **Bewegungs-Spur unsichtbar** – `TrailMarker` ([modules/trailmarker.py](../GestureRecognition/modules/trailmarker.py))
   nutzt einen eigenen Layer `trail` mit demselben Mapping (der Layer wird pro
   Frame zurueckgesetzt, darum muss er explizit gesetzt werden).
3. **`dataset_building` stuerzt bei neuen Aufnahmen ab** –
   `_extract_trajectory` ([labeling.py](../GestureRecognition/labeling.py))
   liest jetzt **beide** Aufnahme-Formate (altes rohes MediaPipe-Objekt **und**
   das neue Dict des aktuellen Detectors) und ueberspringt `None`-Frames (das
   `stop()`-Ergebnis des Detectors wird mitgespeichert).

## Verbesserung: Spur bleibt stehen

Die Spur wird **nicht mehr geloescht, wenn die Hand das Bild verlaesst**. So
sieht man als Aufnehmer den fertig gezeichneten Buchstaben und erkennt sofort,
ob die Aufnahme sauber war (ein wilder Querstrich = Tracking-Sprung = schlecht).
Zurueckgesetzt wird erst, wenn eine neue Geste beginnt. Zusaetzlich wurde
`max_trail_length` in [config.yml](../config.yml) von `50` auf `300` erhoeht,
damit der ganze Buchstabe sichtbar bleibt.

> Hinweis: `webcam.deviceIndex` ist pro Rechner unterschiedlich. Falls das Bild
> schwarz bleibt, den richtigen Index ermitteln (siehe README) und lokal in
> `config.yml` eintragen – diese Aenderung **nicht** committen.

## Tool: Aufnahmen pruefen

`review_recordings.py` prueft alle eigenen Aufnahmen mit **denselben Kriterien
wie `dataset_building`** (Mindestlaenge, keine Tracking-Spruenge, kein NaN) und
verschiebt schlechte optional in `recordings_rejected/` (loescht nichts).

```bash
python review_recordings.py --person <name>              # nur Bericht
python review_recordings.py --person <name> --quarantine # schlechte aussortieren
```

Workflow zum Auffuellen auf z.B. 15 gute pro Buchstabe:

```bash
python collect_alphabet.py --person <name>               # fehlende nachnehmen
python review_recordings.py --person <name> --quarantine # pruefen & aussortieren
# wiederholen, bis "fehlt 0"
```
