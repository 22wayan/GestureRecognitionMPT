from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from GestureRecognition.hmmclassifier import HMMClassifier


def _normalize_dataset(X, y=None):
    if isinstance(X, dict):
        if "sequences" in X and "labels" in X:
            sequences = X["sequences"]
            labels = X["labels"]
        elif "X" in X and "y" in X:
            sequences = X["X"]
            labels = X["y"]
        else:
            sequences = []
            labels = []
            for label, label_sequences in X.items():
                for sequence in label_sequences:
                    sequences.append(sequence)
                    labels.append(label)
    else:
        sequences = X
        labels = y

    if labels is None:
        raise ValueError("Fuer die Grid-Search werden Sequenzen und Labels benoetigt.")

    if len(sequences) != len(labels):
        raise ValueError("Anzahl von Sequenzen und Labels passt nicht zusammen.")

    return list(sequences), [str(label) for label in labels]


def _make_stratified_folds(labels, n_splits=5, random_state=42):
    label_counts = Counter(labels)
    if len(label_counts) < 2:
        raise ValueError("Fuer Cross-Validation werden mindestens zwei Klassen benoetigt.")

    max_splits = min(label_counts.values())
    if max_splits < 2:
        raise ValueError("Jede Klasse braucht mindestens zwei Beispiele fuer CV.")

    n_splits = min(n_splits, max_splits)
    rng = np.random.default_rng(random_state)
    folds = [[] for _ in range(n_splits)]
    grouped_indices = defaultdict(list)

    for index, label in enumerate(labels):
        grouped_indices[label].append(index)

    for label_indices in grouped_indices.values():
        rng.shuffle(label_indices)
        for offset, sample_index in enumerate(label_indices):
            folds[offset % n_splits].append(sample_index)

    return [sorted(fold) for fold in folds]


def _accuracy_score(y_true, y_pred):
    if not y_true:
        return 0.0
    correct = sum(true_label == pred_label for true_label, pred_label in zip(y_true, y_pred))
    return correct / len(y_true)


def _stack_sequences(sequences):
    lengths = [len(np.asarray(sequence)) for sequence in sequences]
    X = np.vstack([np.asarray(sequence, dtype=float) for sequence in sequences])
    return X, lengths


def _evaluate_configuration(
    sequences,
    labels,
    n_components,
    covariance_type,
    n_splits=5,
    random_state=42,
):
    folds = _make_stratified_folds(labels, n_splits=n_splits, random_state=random_state)
    fold_accuracies = []

    for fold_index, test_indices in enumerate(folds):
        train_indices = [index for index in range(len(sequences)) if index not in test_indices]

        train_sequences = [sequences[index] for index in train_indices]
        train_labels = [labels[index] for index in train_indices]
        test_sequences = [sequences[index] for index in test_indices]
        test_labels = [labels[index] for index in test_indices]

        classifier = HMMClassifier(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state + fold_index,
        )

        X_train, train_lengths = _stack_sequences(train_sequences)
        X_test, test_lengths = _stack_sequences(test_sequences)

        classifier.fit(X_train, train_labels, train_lengths)
        predictions = classifier.predict(X_test, test_lengths)
        fold_accuracies.append(_accuracy_score(test_labels, predictions))

    return {
        "n_components": int(n_components),
        "covariance_type": covariance_type,
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "std_accuracy": float(np.std(fold_accuracies)),
        "fold_accuracies": [float(value) for value in fold_accuracies],
    }


def _write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["n_components", "covariance_type", "mean_accuracy", "std_accuracy", "fold_accuracies"]
        )
        for row in rows:
            writer.writerow(
                [
                    row["n_components"],
                    row["covariance_type"],
                    f'{row["mean_accuracy"]:.4f}',
                    f'{row["std_accuracy"]:.4f}',
                    ", ".join(f"{value:.4f}" for value in row["fold_accuracies"]),
                ]
            )


def _write_markdown_report(rows, best_row, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# HMM Modellvergleich",
        "",
        "## Warum Cross-Validation?",
        "",
        "Cross-Validation ist robuster als ein einzelner Train/Test-Split,",
        "weil jede Aufnahme einmal als Testdaten benutzt wird.",
        "Dadurch haengt das Ergebnis weniger vom Zufall einer einzigen Aufteilung ab.",
        "",
        "## Trade-offs bei n_components",
        "",
        "- Wenige Zustaende: einfacher, stabiler, aber oft grober.",
        "- Viele Zustaende: flexibler, aber leichteres Overfitting.",
        "- Deshalb wird hier systematisch ueber mehrere Werte verglichen.",
        "",
        "## Ergebnisse",
        "",
        "| n_components | covariance_type | mean_accuracy | std_accuracy |",
        "| --- | --- | --- | --- |",
    ]

    for row in rows:
        lines.append(
            f'| {row["n_components"]} | {row["covariance_type"]} | '
            f'{row["mean_accuracy"]:.4f} | {row["std_accuracy"]:.4f} |'
        )

    lines.extend(
        [
            "",
            "## Finale Wahl",
            "",
            f'- Gewaehlt wurde `n_components={best_row["n_components"]}` mit '
            f'`covariance_type={best_row["covariance_type"]}`.',
            f'- Die mittlere Accuracy liegt bei `{best_row["mean_accuracy"]:.4f}`.',
            f'- Die Wahl ist einfach begruendet: beste mittlere Accuracy, '
            "bei gleicher Score-Hoehe waere die kleinere Standardabweichung besser.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_best_config(best_row, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "n_components": best_row["n_components"],
        "covariance_type": best_row["covariance_type"],
        "mean_accuracy": best_row["mean_accuracy"],
        "std_accuracy": best_row["std_accuracy"],
    }
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def grid_search_n_components(
    X,
    y=None,
    n_components_values=(2, 3, 4, 5, 6),
    covariance_type="diag",
    n_splits=5,
    random_state=42,
    plot_path="plots/grid_search.png",
    csv_path="reports/grid_search_n_components.csv",
    best_config_path="reports/hmm_best_config.json",
):
    sequences, labels = _normalize_dataset(X, y)
    results = []

    for n_components in n_components_values:
        results.append(
            _evaluate_configuration(
                sequences=sequences,
                labels=labels,
                n_components=n_components,
                covariance_type=covariance_type,
                n_splits=n_splits,
                random_state=random_state,
            )
        )

    results.sort(key=lambda row: row["n_components"])
    best_row = max(results, key=lambda row: (row["mean_accuracy"], -row["std_accuracy"]))

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    x_values = [row["n_components"] for row in results]
    y_values = [row["mean_accuracy"] for row in results]
    y_errors = [row["std_accuracy"] for row in results]
    plt.errorbar(x_values, y_values, yerr=y_errors, marker="o", capsize=4)
    plt.xlabel("n_components")
    plt.ylabel("CV-Accuracy")
    plt.title("Grid Search fuer n_components")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    _write_csv(results, csv_path)
    _save_best_config(best_row, best_config_path)
    return results, best_row


def compare_configurations(
    X,
    y=None,
    covariance_types=("diag", "full"),
    n_components_values=(2, 3, 4, 5, 6),
    n_splits=5,
    random_state=42,
    csv_path="reports/hmm_configuration_results.csv",
    markdown_path="reports/hmm_configuration_report.md",
    best_config_path="reports/hmm_best_config.json",
):
    sequences, labels = _normalize_dataset(X, y)
    results = []

    for covariance_type in covariance_types:
        for n_components in n_components_values:
            results.append(
                _evaluate_configuration(
                    sequences=sequences,
                    labels=labels,
                    n_components=n_components,
                    covariance_type=covariance_type,
                    n_splits=n_splits,
                    random_state=random_state,
                )
            )

    results.sort(key=lambda row: (-row["mean_accuracy"], row["std_accuracy"], row["n_components"]))
    best_row = results[0]

    _write_csv(results, csv_path)
    _write_markdown_report(results, best_row, markdown_path)
    _save_best_config(best_row, best_config_path)
    return results, best_row
