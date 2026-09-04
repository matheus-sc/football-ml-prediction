from dataclasses import dataclass

import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import has_fit_parameter

from .config import MODEL_NAMES, ExperimentConfig
from .data import infer_column_types, load_dataset
from .models import build_preprocessor, get_classifier, needs_scaling

SEARCH_SPACES: dict[str, dict] = {
    "naive_bayes": {
        "classifier__var_smoothing": loguniform(1e-12, 1e-1),
    },
    "svm": {
        "classifier__C": loguniform(1e-2, 1e3),
        "classifier__gamma": loguniform(1e-5, 1e0),
    },
    "xgboost": {
        "classifier__n_estimators": randint(100, 900),
        "classifier__learning_rate": loguniform(5e-3, 3e-1),
        "classifier__max_depth": randint(2, 7),
        "classifier__min_child_weight": randint(1, 20),
        "classifier__subsample": uniform(0.5, 0.5),
        "classifier__colsample_bytree": uniform(0.4, 0.6),
        "classifier__gamma": uniform(0, 5),
        "classifier__reg_lambda": loguniform(1e-1, 5e1),
        "classifier__reg_alpha": loguniform(1e-3, 1e1),
    },
    "mlp_classifier": {
        "classifier__hidden_layer_sizes": [
            (8,),
            (16,),
            (32,),
            (64,),
            (16, 8),
            (32, 16),
            (64, 32),
        ],
        "classifier__alpha": loguniform(1e-5, 1e1),
        "classifier__learning_rate_init": loguniform(1e-4, 1e-1),
        "classifier__batch_size": [16, 32, 64, 128],
    },
}

SEARCH_ITERATIONS = {
    "naive_bayes": 30,
    "svm": 40,
    "xgboost": 60,
    "mlp_classifier": 40,
}


@dataclass(frozen=True)
class TuningResult:
    classifier_name: str
    scenario_name: str
    best_params: dict
    best_score: float
    scoring: str
    n_iter: int
    cv_results_df: pd.DataFrame


def tune_classifier(
    config: ExperimentConfig,
    classifier_name: str | None = None,
    scoring: str = "f1_macro",
    n_iter: int | None = None,
    n_splits: int = 5,
) -> TuningResult:
    classifier_name = classifier_name or config.classifier_name
    n_iter = n_iter or SEARCH_ITERATIONS.get(classifier_name, 30)
    target = config.target

    dataset = load_dataset(target)
    train_df, _ = train_test_split(
        dataset.df,
        test_size=config.test_size,
        stratify=dataset.df[target.target_column],
        shuffle=True,
        random_state=config.random_state,
    )

    numeric_features, categorical_features = infer_column_types(
        train_df,
        dataset.feature_columns,
    )
    classifier = get_classifier(
        classifier_name, target.include_draws, config.random_state
    )
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    numeric_features,
                    categorical_features,
                    scale_numeric_features=needs_scaling(classifier_name),
                ),
            ),
            ("classifier", classifier),
        ]
    )

    X_train = train_df[dataset.feature_columns]
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_df[target.target_column])

    fit_params = {}
    if config.use_sample_weight and has_fit_parameter(classifier, "sample_weight"):
        fit_params["classifier__sample_weight"] = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )

    search = RandomizedSearchCV(
        pipeline,
        SEARCH_SPACES[classifier_name],
        n_iter=n_iter,
        cv=StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=config.random_state
        ),
        scoring=scoring,
        n_jobs=-1,
        random_state=config.random_state,
        refit=False,
    )
    search.fit(X_train, y_train, **fit_params)

    best_params = {
        key.replace("classifier__", ""): value
        for key, value in search.best_params_.items()
    }

    return TuningResult(
        classifier_name=classifier_name,
        scenario_name=target.scenario_name,
        best_params=best_params,
        best_score=float(search.best_score_),
        scoring=scoring,
        n_iter=n_iter,
        cv_results_df=pd.DataFrame(search.cv_results_),
    )


def tune_all(
    config: ExperimentConfig,
    model_names: tuple[str, ...] = MODEL_NAMES,
    scoring: str = "f1_macro",
) -> tuple[pd.DataFrame, list[TuningResult]]:
    results = [tune_classifier(config, name, scoring=scoring) for name in model_names]
    summary_df = pd.DataFrame(
        [
            {
                "cenario": result.scenario_name,
                "classificador": result.classifier_name,
                "metrica": result.scoring,
                "melhor_score_cv": result.best_score,
                "combinacoes_testadas": result.n_iter,
                "melhores_hiperparametros": result.best_params,
            }
            for result in results
        ]
    ).sort_values("melhor_score_cv", ascending=False)

    return summary_df, results
