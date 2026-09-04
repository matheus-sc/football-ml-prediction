from dataclasses import dataclass

import pandas as pd

from .config import TargetConfig


@dataclass(frozen=True)
class DatasetBundle:
    df: pd.DataFrame
    feature_columns: list[str]
    removed_draws_count: int


def infer_feature_columns(df: pd.DataFrame, target: TargetConfig) -> list[str]:
    required_columns = {target.raw_target_column, target.target_column}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        missing_columns_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Colunas alvo não encontradas no CSV: {missing_columns_text}")

    discarded_columns = {
        target.raw_target_column,
        target.target_column,
        *target.excluded_columns,
    }
    return [column for column in df.columns if column not in discarded_columns]


def infer_column_types(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[list[str], list[str]]:
    numeric_features = (
        df[feature_columns].select_dtypes(include="number").columns.tolist()
    )
    categorical_features = [
        column for column in feature_columns if column not in numeric_features
    ]

    return numeric_features, categorical_features


def load_dataset(target: TargetConfig) -> DatasetBundle:
    if not target.dataset_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {target.dataset_path}")

    df = pd.read_csv(target.dataset_path)

    if target.raw_target_column not in df.columns:
        raise ValueError(
            f"Coluna alvo não encontrada no CSV: {target.raw_target_column}"
        )

    raw_classes = set(df[target.raw_target_column].dropna().unique())
    if target.positive_label not in raw_classes:
        raise ValueError(
            f"Classe positiva não encontrada em {target.raw_target_column}: "
            f"{target.positive_label}"
        )

    removed_draws_count = 0
    if not target.include_draws:
        removed_draws_count = int(
            (df[target.raw_target_column] == target.draw_label).sum()
        )
        df = df[df[target.raw_target_column] != target.draw_label].copy()
        df[target.target_column] = df[target.raw_target_column].where(
            df[target.raw_target_column] == target.positive_label,
            target.negative_label,
        )
    else:
        df[target.target_column] = df[target.raw_target_column]

    if df[target.target_column].nunique() < 2:
        raise ValueError("O dataset precisa ter pelo menos duas classes no target.")

    return DatasetBundle(
        df=df,
        feature_columns=infer_feature_columns(df, target),
        removed_draws_count=removed_draws_count,
    )
