import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .evaluation import ExperimentResult


def get_transformed_feature_names(model: Pipeline) -> list[str]:
    preprocessor = model.named_steps["preprocessor"]
    return preprocessor.get_feature_names_out().tolist()


def transform_features(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    transformed = model.named_steps["preprocessor"].transform(X)
    return np.asarray(transformed)


def predict_proba_from_transformed(model: Pipeline, transformed_values: np.ndarray):
    classifier = model.named_steps["classifier"]
    return classifier.predict_proba(transformed_values)


def get_categorical_feature_indexes(feature_names: list[str]) -> list[int]:
    """Índices das colunas que saíram do OneHotEncoder.

    O LimeTabularExplainer trata toda coluna como contínua por padrão, o que produz
    condições sem sentido para variáveis binárias (`cat__estadio_Arena Condá <= 0.00`).
    Informar quais são categóricas faz o LIME reportá-las como presença/ausência.
    """
    return [i for i, name in enumerate(feature_names) if name.startswith("cat__")]


def prepare_interpretability_data(
    result: ExperimentResult,
    shap_background_size: int = 200,
    shap_explain_size: int = 300,
) -> dict:
    model = result.final_model.model
    X_explain_train = result.train_df[result.feature_columns]
    X_explain_test = result.test_df[result.feature_columns]
    X_train_transformed = transform_features(model, X_explain_train)
    X_test_transformed = transform_features(model, X_explain_test)
    feature_names = get_transformed_feature_names(model)
    background_size = min(shap_background_size, len(X_train_transformed))
    explain_size = min(shap_explain_size, len(X_test_transformed))

    return {
        "X_train_transformed": X_train_transformed,
        "X_test_transformed": X_test_transformed,
        "transformed_feature_names": feature_names,
        "categorical_feature_indexes": get_categorical_feature_indexes(feature_names),
        "class_names": [str(class_name) for class_name in result.model_classes],
        "shap_background": shap.sample(X_train_transformed, background_size),
        "shap_explain_data": X_test_transformed[:explain_size],
    }


def shap_feature_importance(shap_values, feature_names: list[str]) -> pd.DataFrame:
    if hasattr(shap_values, "values"):
        values = np.asarray(shap_values.values)
    elif isinstance(shap_values, list):
        values = np.stack(shap_values, axis=-1)
    else:
        values = np.asarray(shap_values)

    if values.ndim == 3:
        mean_abs_values = np.abs(values).mean(axis=(0, 2))
    elif values.ndim == 2:
        mean_abs_values = np.abs(values).mean(axis=0)
    else:
        raise ValueError(f"Formato de SHAP não suportado: {values.shape}")

    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs_values,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def explain_with_shap(result: ExperimentResult, prepared_data: dict) -> pd.DataFrame:
    model = result.final_model.model
    classifier = model.named_steps["classifier"]

    if isinstance(classifier, XGBClassifier):
        shap_explainer = shap.TreeExplainer(classifier)
        shap_values = shap_explainer(prepared_data["shap_explain_data"])
    else:
        shap_explainer = shap.KernelExplainer(
            lambda values: predict_proba_from_transformed(model, values),
            prepared_data["shap_background"],
        )
        shap_values = shap_explainer.shap_values(
            prepared_data["shap_explain_data"],
            nsamples=100,
        )

    return shap_feature_importance(
        shap_values,
        prepared_data["transformed_feature_names"],
    )


def explain_one_with_lime(
    result: ExperimentResult,
    prepared_data: dict,
    example_index: int = 0,
    num_features: int = 10,
) -> pd.DataFrame:
    model = result.final_model.model
    X_test_transformed = prepared_data["X_test_transformed"]
    lime_index = min(example_index, len(X_test_transformed) - 1)
    explainer = LimeTabularExplainer(
        training_data=prepared_data["X_train_transformed"],
        feature_names=prepared_data["transformed_feature_names"],
        categorical_features=prepared_data["categorical_feature_indexes"],
        class_names=prepared_data["class_names"],
        mode="classification",
        random_state=result.config.random_state,
    )
    lime_probabilities = predict_proba_from_transformed(
        model,
        X_test_transformed[[lime_index]],
    )[0]
    predicted_class_index = int(np.argmax(lime_probabilities))
    explanation = explainer.explain_instance(
        data_row=X_test_transformed[lime_index],
        predict_fn=lambda values: predict_proba_from_transformed(model, values),
        num_features=num_features,
        labels=[predicted_class_index],
    )

    lime_explanation_df = pd.DataFrame(
        explanation.as_list(label=predicted_class_index),
        columns=["feature_condition", "contribution"],
    )
    lime_explanation_df.attrs["example_index"] = lime_index
    lime_explanation_df.attrs["real_class"] = result.y_test.iloc[lime_index]
    lime_explanation_df.attrs["predicted_class"] = prepared_data["class_names"][
        predicted_class_index
    ]
    lime_explanation_df.attrs["predicted_probability"] = lime_probabilities[
        predicted_class_index
    ]

    return lime_explanation_df
