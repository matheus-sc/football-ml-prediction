from dataclasses import dataclass

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


@dataclass(frozen=True)
class ModelBuildResult:
    model: Pipeline
    model_y_train: pd.Series
    target_encoder: LabelEncoder | None


SCALE_SENSITIVE_MODELS = frozenset({"naive_bayes", "svm", "mlp_classifier"})


def needs_scaling(classifier_name: str) -> bool:
    return classifier_name in SCALE_SENSITIVE_MODELS


def build_model_registry(random_state: int) -> dict[bool, dict[str, object]]:
    return {
        # Cenário sem empates
        False: {
            "naive_bayes": GaussianNB(var_smoothing=0.046667),
            "svm": SVC(
                kernel="rbf",
                C=26.37334,
                gamma=0.001588,
                probability=True,
                random_state=random_state,
            ),
            "xgboost": XGBClassifier(
                n_estimators=671,
                learning_rate=0.012746,
                max_depth=3,
                min_child_weight=16,
                subsample=0.98271,
                colsample_bytree=0.724869,
                gamma=3.478922,
                reg_lambda=48.849357,
                reg_alpha=0.011049,
                random_state=random_state,
            ),
            "mlp_classifier": MLPClassifier(
                hidden_layer_sizes=(8,),
                activation="relu",
                solver="adam",
                alpha=8.342988,
                learning_rate_init=0.000395,
                batch_size=128,
                max_iter=1200,
                random_state=random_state,
            ),
        },
        # Cenário com empates
        True: {
            "naive_bayes": GaussianNB(var_smoothing=0.046667),
            "svm": SVC(
                kernel="rbf",
                C=0.196743,
                gamma=0.020541,
                probability=True,
                random_state=random_state,
            ),
            "xgboost": XGBClassifier(
                n_estimators=605,
                learning_rate=0.079867,
                max_depth=5,
                min_child_weight=19,
                subsample=0.646605,
                colsample_bytree=0.612943,
                gamma=4.784004,
                reg_lambda=1.623084,
                reg_alpha=0.336782,
                random_state=random_state,
            ),
            "mlp_classifier": MLPClassifier(
                hidden_layer_sizes=(16,),
                activation="relu",
                solver="adam",
                alpha=0.98777,
                learning_rate_init=0.000351,
                batch_size=32,
                max_iter=1200,
                random_state=random_state,
            ),
        },
    }


def get_classifier(
    classifier_name: str,
    include_draws: bool,
    random_state: int,
):
    registry = build_model_registry(random_state)
    try:
        return clone(registry[include_draws][classifier_name])
    except KeyError as error:
        raise ValueError(f"Classificador desconhecido: {classifier_name}") from error


def prepare_target_for_model(
    classifier,
    y_train: pd.Series,
) -> tuple[pd.Series, LabelEncoder | None]:
    if not isinstance(classifier, XGBClassifier):
        return y_train, None

    target_encoder = LabelEncoder()
    encoded_y_train = pd.Series(
        target_encoder.fit_transform(y_train),
        index=y_train.index,
        name=y_train.name,
    )

    if len(target_encoder.classes_) == 2:
        classifier.set_params(objective="binary:logistic", eval_metric="logloss")
    else:
        classifier.set_params(
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=len(target_encoder.classes_),
        )

    return encoded_y_train, target_encoder


def decode_predictions(predictions, target_encoder: LabelEncoder | None):
    if target_encoder is None:
        return predictions

    return target_encoder.inverse_transform(pd.Series(predictions).astype(int))


def get_model_classes(model: Pipeline, target_encoder: LabelEncoder | None):
    if target_encoder is not None:
        return target_encoder.classes_

    return model.classes_


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric_features: bool,
) -> ColumnTransformer:
    transformers = []

    if numeric_features:
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric_features:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps=numeric_steps), numeric_features))

    if categorical_features:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_features))

    if not transformers:
        raise ValueError(
            "Nenhuma coluna de feature foi encontrada para treinar o modelo."
        )

    return ColumnTransformer(transformers=transformers, sparse_threshold=0.0)


def build_model(
    classifier_name: str,
    include_draws: bool,
    random_state: int,
    numeric_features: list[str],
    categorical_features: list[str],
    y_train: pd.Series,
) -> ModelBuildResult:
    scale_numeric_features = needs_scaling(classifier_name)
    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        scale_numeric_features,
    )
    classifier = get_classifier(classifier_name, include_draws, random_state)
    model_y_train, target_encoder = prepare_target_for_model(classifier, y_train)

    return ModelBuildResult(
        model=Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        ),
        model_y_train=model_y_train,
        target_encoder=target_encoder,
    )
