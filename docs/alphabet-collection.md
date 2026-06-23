# Alphabet-Datenerfassung im Team (A–Z)

Dieses Dokument beschreibt die vereinfachte Gesten-Datenerfassung für die 26
Buchstaben des Alphabets: **was** geändert wurde und **wie** ihr es benutzt.

## Ziel

Jedes der vier Teammitglieder nimmt jeden Buchstaben **genau einmal** auf, um
gute Diversität in den Trainingsdaten zu erzeugen. Die Aufnahmen landen direkt
dort, wo das Training liest, sodass danach ohne Umwege ein Datensatz gebaut und
das Modell trainiert werden kann.

## Was wurde gebaut

| Datei | Änderung |
|-------|----------|
| `collect_alphabet.py` (neu, Repo-Wurzel) | Einstiegsskript für den geführten Aufnahme-Durchlauf. |
| `GestureRecognition/labeling.py` | Neue Funktion `collect_alphabet(person, recordings_dir="recordings")`. Aufnahme-Logik einer Einzelaufnahme in `_record_one_take()` extrahiert (von `data_labeling` mitbenutzt). |
| `README.md` | Abschnitt „Alphabet-Sammlung im Team". |

Wichtig: `data_labeling`, `dataset_building`, `clean_recordings` und die
Pipeline bleiben **funktional unverändert** – `data_labeling` nutzt nur denselben
neuen Helfer (gleiches Verhalten wie zuvor).

## Wie benutzen

### 1. Voraussetzungen (einmalig)

Setup wie in der README: venv aktiviert, `requirements.txt` installiert,
`hand_landmarker.task` heruntergeladen, `recordings.zip` entpackt, Webcam-Index
in `config.yml` gesetzt.

### 2. Aufnahme-Durchlauf starten

```bash
source .venv/bin/activate

# macOS: Qt-Pfad setzen, sonst crasht das Fenster
export QT_QPA_PLATFORM_PLUGIN_PATH=$(python -c "import PyQt5, os; print(os.path.join(os.path.dirname(PyQt5.__file__), 'Qt5', 'plugins', 'platforms'))")

python collect_alphabet.py --person <deinName>
# ohne --person wird der Name interaktiv abgefragt
```

Pro Buchstabe:

1. Großanzeige des Buchstabens + Fortschritt (`A  [1/26]`) → **ENTER** drücken.
2. Kamerafenster öffnet sich → Geste ausführen → **Fenster schließen**.
3. Entscheiden: `s` speichern · `v` verwerfen (Buchstabe wiederholen) ·
   `a` abbrechen (ganzen Durchlauf beenden).

### 3. Resume (jederzeit fortsetzbar)

Bereits aufgenommene Buchstaben dieser Person werden beim nächsten Start
**übersprungen**. Ihr könnt also abbrechen und später weitermachen, ohne etwas
zu überschreiben.

### 4. Speicherort & Benennung

```
recordings/
├── A/
│   ├── A-arian.pkl
│   ├── A-wayan.pkl
│   └── ...
├── B/
│   └── ...
└── Z/
```

Format: `recordings/<Buchstabe>/<Buchstabe>-<person>.pkl`. So ist
nachvollziehbar, dass jede Person jeden Buchstaben einmal hat, und
Doppelaufnahmen derselben Person werden verhindert.

Der Personenname darf keine Leerzeichen, Slashes oder Doppelpunkte enthalten
(er wird Teil des Dateinamens) – sonst gibt es eine klare Fehlermeldung.

## Danach: Datensatz bauen & trainieren

Sind alle vier Personen durch:

```bash
python -c "from GestureRecognition.labeling import dataset_building; dataset_building('data/dataset.pkl')"
```

`dataset_building` liest standardmäßig aus `recordings/`, findet also alle neuen
Aufnahmen automatisch. Optional zur Qualitätskontrolle:

```bash
python -c "from GestureRecognition.visualization import visualize_dataset; visualize_dataset()"
```

Training mit dem erzeugten Datensatz:

```python
import pickle
from GestureRecognition.hmmclassifier import HMMClassifier

d = pickle.load(open("data/dataset.pkl", "rb"))
clf = HMMClassifier()
clf.fit(d["X_train"], d["y_train"], d["lengths_train"])
clf.save("data/hmm.pkl")
```
