# Individuelle Teambeiträge

## Einordnung

Dieses Dokument beschreibt, wie wir die Arbeit im Projekt ungefähr aufgeteilt
haben. Grundlage dafür sind die Commits und die geänderten Dateien in unserer
Git-Historie. Die Zuordnung ist deshalb keine minutengenaue Abrechnung, sondern
eine faire Zusammenfassung der erkennbaren Hauptbeiträge.

Wichtig ist uns dabei: Das Projekt ist als gemeinsame Pipeline entstanden.
Mehrere Personen haben dieselben Komponenten getestet, verbessert oder mit
Trainingsdaten versorgt. Eine Person wird unten als Hauptverantwortliche für
einen Bereich genannt, obwohl andere Teammitglieder dort ebenfalls mitgearbeitet
haben können.

Die vier Schwerpunkte sind bewusst ähnlich groß gewählt:

1. Wahrnehmung, Preprocessing und Live-Pipeline
2. Datenerfassung, Datenaufbereitung und HMM-Grundmodell
3. HMM-Inferenz, Hyperparameter und technische Demonstration
4. Evaluation, Robustheit und reproduzierbarer Dataset-Workflow

Zusätzlich haben alle vier Teammitglieder eigene Gesten aufgenommen. Dadurch
enthält der Datensatz unterschiedliche Personen, Geschwindigkeiten und
Ausführungsstile.

## Yannik Huber

### Hauptschwerpunkt: Wahrnehmung, Preprocessing und Live-Integration

Yannik hat vor allem an dem Teil gearbeitet, der aus dem Kamerabild eine
klassifizierbare Gestensequenz macht. Dazu gehören die Handerkennung, die
Fingertrajektorie, die Live-Segmentierung und die Verbindung der Module in der
laufenden Pipeline.

Konkrete Beiträge aus der Git-Historie:

- MediaPipe-Handerkennung im `HandDetector` umgesetzt (`82ddecd`).
- `TrailMarker` zur sichtbaren Darstellung der Fingerbewegung entwickelt
  (`68021c7`).
- `Preprocessor` zum Sammeln und Normalisieren der Trajektorie implementiert
  (`46077cf`).
- Trainings- und Live-Featureformat aufeinander abgestimmt (`ffdf139`).
- Live-Segmentierung mit Hysterese, Pufferbereinigung, Entprellen und größerem
  Puffer stabilisiert (`5b1082d`, `11e62bf`, `ee7c3cc`).
- Klassifikation direkt in das Kamerabild integriert (`58b344f`).
- Aufnahme-Galerie und Visualisierungsworkflow erweitert (`4dc57bc`).
- Training-und-Speichern-Ablauf in `train.py` aufgebaut (`8cb303e`).
- Regressionstest für zusammenhängende Live-Gesten erstellt (`9cf423c`).
- Grid Search später vereinfacht und die Wahl von zehn Zuständen ausgewertet
  (`2299a5b`, `aaedd00`).

### Was Yannik in der Prüfung erklären kann

- Wie MediaPipe aus einem Bild 21 Handlandmarken erzeugt.
- Warum die Zeigefingerspitze als Landmark 8 verfolgt wird.
- Wie eine Geste gestartet und beendet wird.
- Warum Start- und Stoppschwelle eine Hysterese bilden.
- Warum Training und Live-Modus exakt dasselbe Featureformat benötigen.
- Wie Detector, Preprocessor und HMM-Modul in SignalHub zusammenspielen.

## Arian Sharifi-Tabar

### Hauptschwerpunkt: Datenerfassung, Dataset und HMM-Grundmodell

Arian hat vor allem die Grundlage für die systematische Datensammlung und das
erste vollständige HMM-Training geschaffen. Sein Schwerpunkt verbindet die
Rohaufnahmen mit der Datenstruktur, die der Klassifikator benötigt.

Konkrete Beiträge aus der Git-Historie:

- Hilfsfunktionen zum Bereinigen von Rohaufnahmen implementiert (`df91329`).
- Ersten HMM-Klassifikator mit einem `GaussianHMM` pro Klasse umgesetzt
  (`80c149b`).
- `dataset_building()` für `X`, `y` und Sequenzlängen implementiert (`2cc32c0`).
- Erste Visualisierung des Datensatzes entwickelt (`c0a3ca8`).
- Geführte Alphabetaufnahme für A–Z aufgebaut (`9d5ecaf`).
- Aufnahme-Pipeline end-to-end repariert und mehrfaches Aufnehmen pro Buchstabe
  ermöglicht (`beadd44`, `5d6ee28`).
- Review-Rückmeldungen zur Dataset-Erstellung eingearbeitet (`8a22081`).
- Eigene A–Z-Aufnahmen sowie zusätzliche Beispiele für schwache Klassen
  beigesteuert (`f37b4bb`, `cbf4298`, `de3f186`, `0bfb8b7`).

### Was Arian in der Prüfung erklären kann

- Wie Rohaufnahmen nach Labels organisiert werden.
- Warum zu kurze Sequenzen und Tracking-Sprünge entfernt werden.
- Wie unterschiedliche Sequenzlängen für ein HMM gespeichert werden.
- Warum pro Klasse ein eigenes HMM trainiert wird.
- Wie `fit()`, `decision_function()` und `predict()` zusammenarbeiten.
- Wie aus vielen einzelnen Pickle-Aufnahmen ein Trainingsdatensatz entsteht.

## Azad Aygün

### Hauptschwerpunkt: HMM-Inferenz, Hyperparameter und Modellverbesserung

Azad hat sich hauptsächlich mit der praktischen Verwendung und Verbesserung des
HMM-Modells beschäftigt. Dazu gehören die Live-Inferenz, die Suche nach
geeigneten Hyperparametern, das Resampling und die verständliche Darstellung
der finalen Lösung.

Konkrete Beiträge aus der Git-Historie:

- Erste Live-Inferenz mit dem HMM-Modul implementiert (`4eada0f`).
- Grid Search für unterschiedliche HMM-Konfigurationen entwickelt (`66fb7bb`).
- Grid Search an die aktuelle Klassifikator-API angepasst (`4dab636`).
- Review-Rückmeldungen zur Live-Inferenz umgesetzt (`81f6434`).
- Sequenzen auf eine einheitliche Länge gebracht und die Zustandszahl erhöht;
  dadurch stieg die gemessene Accuracy ungefähr von 72 % auf 90 % (`64e7629`).
- README, Designentscheidungen, aktuelle Ergebnisse und Demo-GIFs für die
  Abschlussdarstellung zusammengeführt (`a9718fa`).
- Einen vollständigen eigenen A–Z-Aufnahmesatz beigesteuert (`e01bccf`,
  `fb87af0`, `d59cd98`).

### Was Azad in der Prüfung erklären kann

- Wie die Log-Likelihood eines HMM zur Klassifikation verwendet wird.
- Warum die Klasse mit dem größten Score gewählt wird.
- Warum Resampling Unterschiede im Zeichentempo reduziert.
- Welche Wirkung `n_components`, Kovarianztyp und `min_covar` haben.
- Wie eine Grid Search verschiedene Konfigurationen fair vergleicht.
- Wie das gespeicherte Modell im Live-Modul geladen und verwendet wird.

## Wayan Schmidt

### Hauptschwerpunkt: Evaluation, Robustheit und reproduzierbarer Workflow

Wayan hat vor allem dafür gesorgt, dass die Modellqualität messbar ist und der
Dataset-Workflow reproduzierbar ausgeführt werden kann. Sein Bereich verbindet
Testdaten, Confusion Matrix, numerische Stabilität und Bedienbarkeit.

Konkrete Beiträge aus der Git-Historie:

- `evaluate_classifier()` mit Accuracy, Confusion Matrix und vollständigem
  Personen-Holdout umgesetzt (`67f830f`).
- Numerischen HMM-Kollaps durch eine geeignete `min_covar`-Untergrenze behoben;
  insbesondere wurde die zuvor problematische Klasse F wieder stabil erkannt
  (`edba89e`).
- Kommandozeilen-Wrapper `build_dataset.py` für eine reproduzierbare
  Dataset-Erstellung entwickelt (`ece0cc1`).
- Frühe Projektdokumentation und Abbildungen ins Repository eingebracht
  (`ab3c5d1`, `b9884c7`).
- Eigene Alphabetaufnahmen und gezielte Zusatzaufnahmen für schwache Klassen
  beigesteuert (`879751e`, `15c9272`, `4ec4805`).
- Die Aufnahmen wurden mit dem Review-Workflow geprüft, bevor sie in die
  Evaluation eingeflossen sind.

### Was Wayan in der Prüfung erklären kann

- Warum Trainings- und Testdaten strikt getrennt sein müssen.
- Wie Accuracy und Confusion Matrix berechnet und gelesen werden.
- Warum ein Personen-Holdout realistischer als ein normaler Zufallssplit ist.
- Wie `min_covar` numerische Probleme in Gauß-HMMs verhindert.
- Wie der Datensatz reproduzierbar per CLI erzeugt wird.
- Welche Klassen häufig verwechselt werden und wie man die Daten verbessert.

## Gemeinsam erledigte Aufgaben

Die folgenden Aufgaben ordnen wir bewusst nicht nur einer Person zu:

- Planung der gesamten Pipeline und Abstimmung der Signalnamen.
- Aufnahme von Trainingsdaten durch alle vier Teammitglieder.
- Testen des Live-Modus unter unterschiedlichen Aufnahmebedingungen.
- Besprechen und Beheben von Fehlern zwischen Training und Inferenz.
- Auswertung der Confusion Matrix und Auswahl schwacher Klassen.
- Review von Pull Requests und gemeinsames Zusammenführen der Teilbereiche.
- Vorbereitung der Demonstration und Erklärung für die Prüfung.

## Warum die Verteilung fair ist

Jeder Schwerpunkt enthält mehrere Arten von Arbeit:

- eine technische Kernkomponente,
- Integration mit mindestens einem anderen Projektteil,
- Tests oder Evaluation,
- Dokumentation beziehungsweise Erklärung,
- eigene Trainingsdaten.

Die Anzahl der Commits ist dabei nicht allein entscheidend. Ein einzelner
Datensatz-Commit kann hunderte aufwendig aufgenommene Dateien enthalten, während
eine Fehlerbehebung aus mehreren kleinen Commits bestehen kann. Deshalb bewerten
wir Verantwortung, Schwierigkeit und Nutzen für das Gesamtsystem gemeinsam und
nicht nur die reine Commit-Anzahl.

## Kurze Prüfungsaufteilung

Falls wir das Projekt in der Prüfung gemeinsam vorstellen, bietet sich folgende
gleichmäßige Reihenfolge an:

1. **Yannik:** Kamera, Landmarken, Preprocessing und Live-Segmentierung.
2. **Arian:** Aufnahme, Bereinigung, Dataset-Struktur und HMM-Grundprinzip.
3. **Azad:** HMM-Inferenz, Resampling, Hyperparameter und Grid Search.
4. **Wayan:** Evaluation, Confusion Matrix, Personen-Holdout und Robustheit.

Zum Abschluss erklärt jede Person kurz, welche eigenen Aufnahmen und Tests sie
beigesteuert hat und welche Verbesserung sie aus den Ergebnissen ableiten würde.
