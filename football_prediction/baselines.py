"""Linhas de base para dar significado às métricas dos modelos.

Sem uma referência explícita, uma acurácia isolada não diz nada: no cenário binário
`fora` vs `nao_fora` desta base, prever sempre `nao_fora` já entrega 75,6%. As três
referências implementadas aqui são:

1. `majority`   — sempre a classe mais frequente do treino. É o piso da acurácia.
2. `odds`       — argmax da probabilidade implícita das casas de apostas.
3. `odds_prior` — a mesma probabilidade dividida pelo prior das classes, que é a
                  regra de decisão de custo balanceado. É o piso da acurácia
                  balanceada e do macro-F1.

A comparação com (2) e (3) responde à pergunta que nenhum dos trabalhos relacionados
faz: o modelo aprende algo além do que o mercado de apostas já precificou?
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)

from .evaluation import ExperimentResult

ODDS_COLUMNS = ("odds_mandante_vence", "odds_empate", "odds_visitante_vence")
ODDS_CLASSES = ("casa", "empate", "fora")


def implied_probabilities(df: pd.DataFrame) -> np.ndarray:
    """Probabilidades implícitas das odds, normalizadas para somar 1.

    A soma de 1/odds excede 1 porque embute a margem da casa (o overround, ~6,5%
    nesta base). Normalizar remove essa margem e devolve probabilidades comparáveis
    com as que os modelos produzem.
    """
    odds = df.loc[:, list(ODDS_COLUMNS)].replace(0, np.nan).to_numpy(dtype=float)
    inverse = np.divide(1.0, odds, out=np.full_like(odds, np.nan), where=odds > 0)
    totals = np.nansum(inverse, axis=1, keepdims=True)

    return np.divide(
        inverse,
        totals,
        out=np.full_like(inverse, np.nan),
        where=totals > 0,
    )


def _resolve_labels(
    result: ExperimentResult, class_predictions: np.ndarray
) -> np.ndarray:
    """Converte casa/empate/fora para os rótulos do cenário (binário ou não)."""
    target = result.config.target

    if target.include_draws:
        return class_predictions

    return np.where(
        class_predictions == target.positive_label,
        target.positive_label,
        target.negative_label,
    )


def baseline_predictions(result: ExperimentResult, strategy: str) -> pd.Series:
    target = result.config.target
    test_df = result.test_df
    train_labels = result.train_df[target.target_column]

    if strategy == "majority":
        predictions = np.full(len(test_df), train_labels.value_counts().idxmax())
        return pd.Series(
            predictions, index=result.y_test.index, name="vencedor_previsto"
        )

    missing_columns = set(ODDS_COLUMNS).difference(test_df.columns)
    if missing_columns:
        raise ValueError(
            "Colunas de odds ausentes no dataset: " + ", ".join(sorted(missing_columns))
        )

    probabilities = implied_probabilities(test_df)
    train_prior = (
        result.train_df[target.raw_target_column]
        .value_counts(normalize=True)
        .reindex(ODDS_CLASSES)
        .to_numpy()
    )

    if strategy == "odds_prior":
        probabilities = probabilities / train_prior
    elif strategy != "odds":
        raise ValueError(f"Estratégia desconhecida: {strategy}")

    incomplete = ~np.isfinite(probabilities).all(axis=1)
    probabilities[incomplete] = train_prior

    class_predictions = np.asarray(ODDS_CLASSES)[probabilities.argmax(axis=1)]

    return pd.Series(
        _resolve_labels(result, class_predictions),
        index=result.y_test.index,
        name="vencedor_previsto",
    )


def compare_with_baselines(
    result: ExperimentResult,
    strategies: tuple[str, ...] = ("majority", "odds", "odds_prior"),
) -> pd.DataFrame:
    """Métricas do modelo e das linhas de base sobre o mesmo conjunto de teste."""
    target = result.config.target
    names = {
        "majority": "Baseline: classe majoritária",
        "odds": "Baseline: argmax das odds",
        "odds_prior": "Baseline: odds ajustadas pelo prior",
    }

    rows = []
    candidates = [(f"Modelo: {result.config.classifier_name}", result.predictions)]
    candidates += [(names[s], baseline_predictions(result, s)) for s in strategies]

    for name, predictions in candidates:
        report = classification_report(
            result.y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        )
        positive_report = report.get(target.positive_label, {})
        rows.append(
            {
                "abordagem": name,
                "acuracia": accuracy_score(result.y_test, predictions),
                "balanced_accuracy": balanced_accuracy_score(
                    result.y_test, predictions
                ),
                "macro_f1": f1_score(result.y_test, predictions, average="macro"),
                f"{target.positive_label}_precision": positive_report.get(
                    "precision", 0.0
                ),
                f"{target.positive_label}_recall": positive_report.get("recall", 0.0),
                f"{target.positive_label}_f1": positive_report.get("f1-score", 0.0),
            }
        )

    return pd.DataFrame(rows)
