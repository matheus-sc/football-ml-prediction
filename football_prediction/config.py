from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TargetConfig:
    dataset_path: Path = Path("brasileirao_serie_a_2018_2023_v3.csv")
    raw_target_column: str = "vencedor"
    target_column: str = "vencedor_modelado"
    positive_label: str = "fora"
    draw_label: str = "empate"
    include_draws: bool = True
    excluded_columns: tuple[str, ...] = ("data",)

    @property
    def negative_label(self) -> str:
        return f"nao_{self.positive_label}"

    @property
    def scenario_name(self) -> str:
        return "com empates" if self.include_draws else "sem empates"


@dataclass(frozen=True)
class ExperimentConfig:
    target: TargetConfig = TargetConfig()
    classifier_name: str = "xgboost"
    test_size: float = 0.3
    n_splits: int = 5
    n_repeats: int = 5
    random_state: int = 42
    use_sample_weight: bool = True


MODEL_NAMES = (
    "naive_bayes",
    "svm",
    "xgboost",
    "mlp_classifier",
)

THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)

HYPERPARAMETER_KEYS = (
    "var_smoothing",
    "kernel",
    "C",
    "gamma",
    "n_estimators",
    "learning_rate",
    "max_depth",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "reg_lambda",
    "reg_alpha",
    "hidden_layer_sizes",
    "activation",
    "solver",
    "alpha",
    "learning_rate_init",
    "batch_size",
    "max_iter",
)
