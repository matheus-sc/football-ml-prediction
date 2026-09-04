from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import has_fit_parameter

from .config import ExperimentConfig
from .data import infer_column_types, load_dataset
from .models import build_model, decode_predictions, get_model_classes


@dataclass(frozen=True)
class FinalModelResult:
    model: Pipeline
    y_test: pd.Series
    predictions: pd.Series
    prediction_examples: pd.DataFrame
    model_classes: list[str]


@dataclass(frozen=True)
class ExperimentResult:
    config: ExperimentConfig
    removed_draws_count: int
    feature_columns: list[str]
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    fold_results_df: pd.DataFrame
    y_validation_all: pd.Series
    validation_predictions: pd.Series
    validation_model_classes: list[str]
    final_model: FinalModelResult

    @property
    def y_test(self) -> pd.Series:
        return self.final_model.y_test

    @property
    def predictions(self) -> pd.Series:
        return self.final_model.predictions

    @property
    def model_classes(self) -> list[str]:
        return self.final_model.model_classes


def make_fit_params(
    model: Pipeline,
    y_train: pd.Series,
    use_sample_weight: bool,
) -> dict:
    classifier = model.named_steps["classifier"]

    if not use_sample_weight or not has_fit_parameter(classifier, "sample_weight"):
        return {}

    return {
        "classifier__sample_weight": compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )
    }


def evaluate_k_fold(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    target = config.target
    X = train_df[feature_columns]
    y = train_df[target.target_column]
    splitter = RepeatedStratifiedKFold(
        n_splits=config.n_splits,
        n_repeats=config.n_repeats,
        random_state=config.random_state,
    )

    fold_results = []
    y_true_parts = []
    prediction_parts = []
    final_model_classes = None

    for split_number, (train_index, validation_index) in enumerate(
        splitter.split(X, y),
        start=1,
    ):
        repetition = (split_number - 1) // config.n_splits + 1
        fold_number = (split_number - 1) % config.n_splits + 1
        fold_train_df = train_df.iloc[train_index].copy()
        validation_df = train_df.iloc[validation_index].copy()
        numeric_features, categorical_features = infer_column_types(
            fold_train_df,
            feature_columns,
        )

        X_train = fold_train_df[feature_columns]
        y_train = fold_train_df[target.target_column]
        X_validation = validation_df[feature_columns]
        y_validation = validation_df[target.target_column]

        model_result = build_model(
            classifier_name=config.classifier_name,
            include_draws=target.include_draws,
            random_state=config.random_state,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            y_train=y_train,
        )
        model_result.model.fit(
            X_train,
            model_result.model_y_train,
            **make_fit_params(model_result.model, y_train, config.use_sample_weight),
        )

        fold_predictions = decode_predictions(
            model_result.model.predict(X_validation),
            model_result.target_encoder,
        )
        model_classes = get_model_classes(model_result.model, model_result.target_encoder)
        final_model_classes = list(model_classes)

        fold_results.append(
            {
                "repeticao": repetition,
                "fold": fold_number,
                "exemplos_treino": len(X_train),
                "exemplos_validacao": len(X_validation),
                "acuracia": accuracy_score(y_validation, fold_predictions),
                "balanced_accuracy": balanced_accuracy_score(
                    y_validation,
                    fold_predictions,
                ),
                "macro_f1": f1_score(y_validation, fold_predictions, average="macro"),
            }
        )

        if repetition == 1:
            y_true_parts.append(y_validation)
            prediction_parts.append(
                pd.Series(
                    fold_predictions,
                    index=y_validation.index,
                    name="vencedor_previsto",
                )
            )

    if final_model_classes is None:
        raise RuntimeError("Nenhum fold foi avaliado.")

    return (
        pd.DataFrame(fold_results),
        pd.concat(y_true_parts).sort_index(),
        pd.concat(prediction_parts).sort_index(),
        final_model_classes,
    )


def fit_and_evaluate_final_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    config: ExperimentConfig,
) -> FinalModelResult:
    target = config.target
    numeric_features, categorical_features = infer_column_types(train_df, feature_columns)
    X_train = train_df[feature_columns]
    y_train = train_df[target.target_column]
    X_test = test_df[feature_columns]
    y_test = test_df[target.target_column]

    model_result = build_model(
        classifier_name=config.classifier_name,
        include_draws=target.include_draws,
        random_state=config.random_state,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        y_train=y_train,
    )
    model_result.model.fit(
        X_train,
        model_result.model_y_train,
        **make_fit_params(model_result.model, y_train, config.use_sample_weight),
    )

    test_predictions = decode_predictions(
        model_result.model.predict(X_test),
        model_result.target_encoder,
    )
    model_classes = list(get_model_classes(model_result.model, model_result.target_encoder))
    prediction_examples = X_test.copy()
    prediction_examples["resultado_real"] = test_df[target.raw_target_column].values
    prediction_examples["vencedor_real"] = y_test.values
    prediction_examples["vencedor_previsto"] = test_predictions

    if hasattr(model_result.model, "predict_proba"):
        probabilities = model_result.model.predict_proba(X_test)
        for index, class_name in enumerate(model_classes):
            prediction_examples[f"prob_{class_name}"] = probabilities[:, index]

    return FinalModelResult(
        model=model_result.model,
        y_test=y_test,
        predictions=pd.Series(
            test_predictions,
            index=y_test.index,
            name="vencedor_previsto",
        ),
        prediction_examples=prediction_examples,
        model_classes=model_classes,
    )


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    dataset = load_dataset(config.target)
    train_df, test_df = train_test_split(
        dataset.df,
        test_size=config.test_size,
        stratify=dataset.df[config.target.target_column],
        shuffle=True,
        random_state=config.random_state,
    )
    fold_results_df, y_validation_all, validation_predictions, validation_model_classes = (
        evaluate_k_fold(train_df, dataset.feature_columns, config)
    )
    final_model = fit_and_evaluate_final_model(
        train_df,
        test_df,
        dataset.feature_columns,
        config,
    )

    return ExperimentResult(
        config=config,
        removed_draws_count=dataset.removed_draws_count,
        feature_columns=dataset.feature_columns,
        train_df=train_df,
        test_df=test_df,
        fold_results_df=fold_results_df,
        y_validation_all=y_validation_all,
        validation_predictions=validation_predictions,
        validation_model_classes=validation_model_classes,
        final_model=final_model,
    )


def metric_summary(
    y_true: pd.Series,
    predictions: pd.Series,
) -> dict[str, float]:
    return {
        "acuracia": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "macro_f1": f1_score(y_true, predictions, average="macro"),
    }


def print_experiment_summary(result: ExperimentResult) -> None:
    config = result.config
    validation_metrics = metric_summary(
        result.y_validation_all,
        result.validation_predictions,
    )
    test_metrics = metric_summary(result.y_test, result.predictions)

    print(f"Modelo: {config.classifier_name}")
    print(f"Dataset: {config.target.dataset_path.name}")
    print(f"Target original: {config.target.raw_target_column}")
    print(f"Target modelado: {config.target.target_column}")
    print(f"Split: {1 - config.test_size:.0%} treino / {config.test_size:.0%} teste final")
    print(f"Cenário: {config.target.scenario_name}")
    print(f"Empates removidos: {result.removed_draws_count}")
    print(f"Colunas descartadas: {', '.join(config.target.excluded_columns) or 'nenhuma'}")
    print(f"Features usadas: {len(result.feature_columns)}")
    print(f"Exemplos de treino: {len(result.train_df)}")
    print(f"Exemplos de teste final: {len(result.test_df)}")
    print(
        f"Avaliações no treino: {len(result.fold_results_df)} "
        f"({config.n_splits} folds x {config.n_repeats} repetições)"
    )
    print(f"Classes: {result.model_classes}")

    print("\nValidação cruzada repetida no treino (média ± desvio entre folds):")
    print(
        f"Acurácia média: {result.fold_results_df['acuracia'].mean():.4f} "
        f"± {result.fold_results_df['acuracia'].std():.4f}"
    )
    print(
        "Balanced accuracy média: "
        f"{result.fold_results_df['balanced_accuracy'].mean():.4f} "
        f"± {result.fold_results_df['balanced_accuracy'].std():.4f}"
    )
    print(
        f"Macro-F1 médio: {result.fold_results_df['macro_f1'].mean():.4f} "
        f"± {result.fold_results_df['macro_f1'].std():.4f}"
    )
    print(f"Acurácia out-of-fold agregada: {validation_metrics['acuracia']:.4f}")
    print(
        "Balanced accuracy out-of-fold agregada: "
        f"{validation_metrics['balanced_accuracy']:.4f}"
    )
    print(f"Macro-F1 out-of-fold agregado: {validation_metrics['macro_f1']:.4f}")

    print("\nTeste final reservado:")
    print(f"Acurácia: {test_metrics['acuracia']:.4f}")
    print(f"Balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"Macro-F1: {test_metrics['macro_f1']:.4f}")


def confusion_matrix_df(result: ExperimentResult) -> pd.DataFrame:
    return pd.DataFrame(
        confusion_matrix(
            result.y_test,
            result.predictions,
            labels=result.model_classes,
        ),
        index=result.model_classes,
        columns=result.model_classes,
    )


def classification_report_text(result: ExperimentResult) -> str:
    return classification_report(result.y_test, result.predictions, digits=4)
