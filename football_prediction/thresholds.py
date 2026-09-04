import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
)

from .config import THRESHOLDS
from .evaluation import ExperimentResult


def predict_with_threshold(
    probabilities: np.ndarray,
    model_classes: list[str],
    target_class: str,
    threshold: float,
    index,
) -> pd.Series:
    target_index = model_classes.index(target_class)
    other_indexes = [i for i in range(len(model_classes)) if i != target_index]
    fallback = [model_classes[i] for i in np.asarray(probabilities)[:, other_indexes].argmax(axis=1)]

    return pd.Series(
        np.where(probabilities[:, target_index] >= threshold, target_class, fallback),
        index=index,
        name="vencedor_previsto",
    )


def target_class_probabilities(result: ExperimentResult, target_class: str) -> np.ndarray:
    model = result.final_model.model

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            f"O classificador {result.config.classifier_name} não fornece predict_proba."
        )

    if target_class not in result.model_classes:
        raise ValueError(
            f"Classe {target_class} não está entre as classes do modelo: "
            f"{result.model_classes}"
        )

    return model.predict_proba(result.test_df[result.feature_columns])


def evaluate_thresholds(
    result: ExperimentResult,
    thresholds: tuple[float, ...] = THRESHOLDS,
    target_class: str | None = None,
) -> pd.DataFrame:
    target_class = target_class or result.config.target.positive_label
    probabilities = target_class_probabilities(result, target_class)
    y_test = result.y_test

    rows = []
    for threshold in thresholds:
        predictions = predict_with_threshold(
            probabilities,
            result.model_classes,
            target_class,
            threshold,
            y_test.index,
        )
        report = classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        )
        class_report = report.get(target_class, {})
        predicted_as_target = predictions == target_class
        is_target = y_test == target_class

        rows.append(
            {
                f"limiar_{target_class}": threshold,
                "acuracia": accuracy_score(y_test, predictions),
                "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
                "macro_f1": f1_score(y_test, predictions, average="macro"),
                f"{target_class}_precision": class_report.get("precision", 0.0),
                f"{target_class}_recall": class_report.get("recall", 0.0),
                f"{target_class}_f1": class_report.get("f1-score", 0.0),
                f"verdadeiros_{target_class}": int((predicted_as_target & is_target).sum()),
                f"falsos_{target_class}": int((predicted_as_target & ~is_target).sum()),
                f"{target_class}_perdidos": int((~predicted_as_target & is_target).sum()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["macro_f1", f"{target_class}_f1"],
        ascending=[False, False],
    )


def precision_recall_frontier(
    result: ExperimentResult,
    target_class: str | None = None,
) -> pd.DataFrame:
    """Fronteira completa de precisão x revocação da classe de interesse.

    É o gráfico que permite comparar este trabalho com um ponto de operação
    publicado por outro autor sem depender da escolha de limiar de cada um.
    """
    target_class = target_class or result.config.target.positive_label
    probabilities = target_class_probabilities(result, target_class)
    target_index = result.model_classes.index(target_class)
    scores = probabilities[:, target_index]
    y_binary = (result.y_test == target_class).astype(int)

    precision, recall, thresholds = precision_recall_curve(y_binary, scores)
    denominator = precision[:-1] + recall[:-1]

    return pd.DataFrame(
        {
            "limiar": thresholds,
            "precision": precision[:-1],
            "recall": recall[:-1],
            "f1": np.divide(
                2 * precision[:-1] * recall[:-1],
                denominator,
                out=np.zeros_like(denominator),
                where=denominator > 0,
            ),
        }
    )


def precision_at_recall(frontier: pd.DataFrame, recall_target: float) -> pd.Series:
    """Precisão deste modelo na revocação em que outro trabalho opera."""
    return frontier.iloc[(frontier["recall"] - recall_target).abs().argmin()]
