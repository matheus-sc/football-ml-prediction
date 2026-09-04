from dataclasses import replace

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .config import HYPERPARAMETER_KEYS, MODEL_NAMES, ExperimentConfig, TargetConfig
from .evaluation import ExperimentResult, run_experiment
from .models import get_classifier


def summarize_hyperparameters(classifier) -> dict:
    params = classifier.get_params()
    return {key: params[key] for key in HYPERPARAMETER_KEYS if key in params}


def summarize_experiment(result: ExperimentResult) -> dict:
    target = result.config.target
    report = classification_report(
        result.y_test,
        result.predictions,
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(
        result.y_test,
        result.predictions,
        labels=result.model_classes,
    )
    summary = {
        "cenario": target.scenario_name,
        "classificador": result.config.classifier_name,
        "classes": ", ".join(result.model_classes),
        "empates_removidos": result.removed_draws_count,
        "treino": len(result.train_df),
        "teste": len(result.test_df),
        "cv_acuracia_media": result.fold_results_df["acuracia"].mean(),
        "cv_acuracia_std": result.fold_results_df["acuracia"].std(),
        "cv_balanced_accuracy_media": result.fold_results_df[
            "balanced_accuracy"
        ].mean(),
        "cv_macro_f1_medio": result.fold_results_df["macro_f1"].mean(),
        "teste_acuracia": accuracy_score(result.y_test, result.predictions),
        "teste_balanced_accuracy": balanced_accuracy_score(
            result.y_test,
            result.predictions,
        ),
        "teste_macro_f1": f1_score(
            result.y_test,
            result.predictions,
            average="macro",
        ),
        "hiperparametros": summarize_hyperparameters(
            get_classifier(
                result.config.classifier_name,
                target.include_draws,
                result.config.random_state,
            )
        ),
    }

    for class_name in result.model_classes:
        class_report = report[class_name]
        class_index = result.model_classes.index(class_name)
        summary[f"{class_name}_precision"] = class_report["precision"]
        summary[f"{class_name}_recall"] = class_report["recall"]
        summary[f"{class_name}_f1"] = class_report["f1-score"]
        summary[f"{class_name}_suporte"] = class_report["support"]
        summary[f"verdadeiros_{class_name}"] = confusion[class_index, class_index]

    return summary


def compare_models_by_scenario(
    base_config: ExperimentConfig,
    model_names: tuple[str, ...] = MODEL_NAMES,
    include_draw_options: tuple[bool, ...] = (True, False),
) -> tuple[pd.DataFrame, pd.DataFrame, list[ExperimentResult]]:
    experiment_results = []
    summaries = []

    for include_draws in include_draw_options:
        target = replace(base_config.target, include_draws=include_draws)
        for classifier_name in model_names:
            config = replace(
                base_config,
                target=target,
                classifier_name=classifier_name,
            )
            result = run_experiment(config)
            experiment_results.append(result)
            summaries.append(summarize_experiment(result))

    results_df = pd.DataFrame(summaries)
    summary_df = results_df.drop(columns="hiperparametros").sort_values(
        ["cenario", "teste_macro_f1"],
        ascending=[True, False],
    )
    hyperparameters_df = results_df[
        ["cenario", "classificador", "hiperparametros"]
    ].sort_values(["cenario", "classificador"])

    return summary_df, hyperparameters_df, experiment_results


def make_config(
    classifier_name: str = "xgboost",
    include_draws: bool = True,
    random_state: int = 42,
    use_sample_weight: bool = True,
) -> ExperimentConfig:
    return ExperimentConfig(
        classifier_name=classifier_name,
        random_state=random_state,
        use_sample_weight=use_sample_weight,
        target=TargetConfig(include_draws=include_draws),
    )
