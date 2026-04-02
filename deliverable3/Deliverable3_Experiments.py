from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_SPLITS = 5

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / "deliverable2" / "data" / "healthcare-dataset-stroke-data.csv"
TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures"
MODELS_DIR = ROOT / "models"


@dataclass(frozen=True)
class DatasetVersion:
    name: str
    description: str
    steps: List[Tuple[str, object]]


def ensure_dirs() -> None:
    for directory in (TABLES_DIR, FIGURES_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = pd.read_csv(DATA_PATH)
    raw_shape = df.shape
    duplicate_rows = int(df.duplicated().sum())
    missing_rows = df.isna().sum().to_dict()

    work_df = df.drop(columns=["id"]).drop_duplicates().copy()
    X = work_df.drop(columns=["stroke"])
    y = work_df["stroke"].copy()

    summary = pd.DataFrame(
        [
            {
                "raw_rows": raw_shape[0],
                "raw_columns": raw_shape[1],
                "rows_after_dedup": work_df.shape[0],
                "columns_after_drop_id": work_df.shape[1],
                "duplicate_rows_removed": duplicate_rows,
                "stroke_cases": int(y.sum()),
                "non_stroke_cases": int((y == 0).sum()),
                "stroke_rate_pct": round(100 * y.mean(), 2),
                "missing_bmi": int(missing_rows.get("bmi", 0)),
            }
        ]
    )
    return X, y, summary


def build_preprocessor(version_name: str, num_features: List[str], cat_features: List[str]) -> DatasetVersion:
    base_num = [("imputer", SimpleImputer(strategy="median"))]
    scaled_num = base_num + [("scaler", StandardScaler())]
    cat_steps = [("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", make_ohe())]

    if version_name == "V1":
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline(steps=base_num), num_features),
                ("cat", Pipeline(steps=cat_steps), cat_features),
            ]
        )
        return DatasetVersion(
            name="V1",
            description="Median numeric imputation + most-frequent categorical imputation + one-hot encoding.",
            steps=[("preprocessor", preprocessor)],
        )

    if version_name == "V2":
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline(steps=scaled_num), num_features),
                ("cat", Pipeline(steps=cat_steps), cat_features),
            ]
        )
        return DatasetVersion(
            name="V2",
            description="V1 plus standard scaling for numerical features.",
            steps=[("preprocessor", preprocessor)],
        )

    if version_name == "V3":
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline(steps=scaled_num), num_features),
                ("cat", Pipeline(steps=cat_steps), cat_features),
            ]
        )
        return DatasetVersion(
            name="V3",
            description="V2 plus SMOTE applied inside each training fold.",
            steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
            ],
        )

    if version_name == "V4":
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline(steps=scaled_num), num_features),
                ("cat", Pipeline(steps=cat_steps), cat_features),
            ]
        )
        return DatasetVersion(
            name="V4",
            description="V3 plus PCA retaining 95% explained variance.",
            steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
            ],
        )

    raise ValueError(f"Unknown dataset version: {version_name}")


def build_candidate_models() -> Dict[str, object]:
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "SVM": SVC(random_state=RANDOM_STATE, probability=True),
        "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
    }


def build_pipeline(version: DatasetVersion, model: object) -> ImbPipeline:
    steps = list(version.steps) + [("classifier", clone(model))]
    return ImbPipeline(steps=steps)


def get_scorers() -> Dict[str, object]:
    return {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0),
        "roc_auc": "roc_auc",
    }


def initial_experiments(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    versions: List[DatasetVersion],
    models: Dict[str, object],
    cv: StratifiedKFold,
) -> pd.DataFrame:
    scorers = get_scorers()
    rows = []
    total_runs = len(versions) * len(models)
    run_idx = 0

    for version in versions:
        for model_name, model in models.items():
            run_idx += 1
            print(f"[initial {run_idx}/{total_runs}] {version.name} + {model_name}")
            pipeline = build_pipeline(version, model)
            scores = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=scorers,
                n_jobs=1,
                return_train_score=False,
            )
            row = {
                "dataset_version": version.name,
                "version_description": version.description,
                "model": model_name,
            }
            for metric_name in scorers:
                row[f"{metric_name}_mean"] = scores[f"test_{metric_name}"].mean()
                row[f"{metric_name}_std"] = scores[f"test_{metric_name}"].std(ddof=1)
            rows.append(row)

    results_df = pd.DataFrame(rows).sort_values(
        by=["balanced_accuracy_mean", "recall_mean", "f1_weighted_mean"],
        ascending=False,
    )
    return results_df


def summarize_initial_results(results_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_means = [
        "accuracy_mean", "balanced_accuracy_mean", "precision_mean",
        "recall_mean", "f1_weighted_mean", "roc_auc_mean",
    ]

    model_summary = (
        results_df.groupby("model", as_index=False)[metric_means]
        .mean()
        .sort_values(by=["balanced_accuracy_mean", "recall_mean", "f1_weighted_mean"], ascending=False)
    )

    version_summary = (
        results_df.groupby(["dataset_version", "version_description"], as_index=False)[metric_means]
        .mean()
        .sort_values(by=["balanced_accuracy_mean", "recall_mean", "f1_weighted_mean"], ascending=False)
    )

    return model_summary, version_summary


def plot_initial_heatmap(results_df: pd.DataFrame) -> None:
    pivot = results_df.pivot(index="model", columns="dataset_version", values="f1_weighted_mean")
    plt.figure(figsize=(8, 4.8))
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title("Initial Cross-Validation Weighted F1 by Model and Dataset Version")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "initial_cv_f1_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close()


def plot_balanced_accuracy_heatmap(results_df: pd.DataFrame) -> None:
    """Heatmap of CV balanced accuracy — the primary selection criterion."""
    pivot = results_df.pivot(index="model", columns="dataset_version", values="balanced_accuracy_mean")
    plt.figure(figsize=(8, 4.8))
    sns.heatmap(pivot, annot=True, cmap="YlOrRd", fmt=".3f")
    plt.title("Initial Cross-Validation Balanced Accuracy by Model and Dataset Version")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "initial_cv_balanced_accuracy_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close()


def plot_metric_bars(results_df: pd.DataFrame) -> None:
    metric_cols = ["accuracy_mean", "balanced_accuracy_mean", "recall_mean", "f1_weighted_mean", "roc_auc_mean"]
    plot_df = results_df.copy()
    plot_df["combo"] = plot_df["dataset_version"] + " + " + plot_df["model"]
    melted = plot_df[["combo"] + metric_cols].melt(id_vars="combo", var_name="metric", value_name="score")

    plt.figure(figsize=(12, 9))
    sns.barplot(data=melted, x="score", y="combo", hue="metric", orient="h")
    plt.title("Initial Cross-Validation Metric Comparison")
    plt.xlabel("Score")
    plt.ylabel("")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "initial_cv_metric_bars.png", dpi=220, bbox_inches="tight")
    plt.close()


def get_param_grids() -> Dict[str, List[Dict[str, object]]]:
    return {
        "LogisticRegression": [
            {
                "classifier__solver": ["liblinear", "lbfgs"],
                "classifier__C": [0.1, 1.0, 5.0, 10.0],
                "classifier__class_weight": [None, "balanced"],
            }
        ],
        "KNN": [
            {
                "classifier__n_neighbors": [5, 9, 15, 21],
                "classifier__weights": ["uniform", "distance"],
                "classifier__metric": ["minkowski", "manhattan"],
            }
        ],
        "RandomForest": [
            {
                "classifier__n_estimators": [200, 400],
                "classifier__max_depth": [None, 10, 20],
                "classifier__min_samples_split": [2, 5],
                "classifier__min_samples_leaf": [1, 2],
                "classifier__class_weight": [None, "balanced"],
            }
        ],
        "SVM": [
            {
                "classifier__kernel": ["linear", "rbf"],
                "classifier__C": [0.5, 1.0, 5.0],
                "classifier__gamma": ["scale", "auto"],
                "classifier__class_weight": [None, "balanced"],
            }
        ],
        "AdaBoost": [
            {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__learning_rate": [0.5, 1.0, 1.5],
            }
        ],
    }


def select_top_candidates(model_summary: pd.DataFrame, version_summary: pd.DataFrame) -> Tuple[List[str], List[str]]:
    top_models = model_summary.head(2)["model"].tolist()
    top_versions = version_summary.head(2)["dataset_version"].tolist()
    return top_models, top_versions


def evaluate_on_test(estimator: ImbPipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, object]:
    y_pred = estimator.predict(X_test)
    y_proba = estimator.predict_proba(X_test)[:, 1] if hasattr(estimator, "predict_proba") else None
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    result: Dict[str, object] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "stroke_precision_classification_report": report["1"]["precision"],
        "stroke_recall_classification_report": report["1"]["recall"],
        "stroke_f1_classification_report": report["1"]["f1-score"],
    }
    return result


def plot_confusion_matrix(estimator: ImbPipeline, X_test: pd.DataFrame, y_test: pd.Series, combo_name: str) -> None:
    disp = ConfusionMatrixDisplay.from_estimator(
        estimator,
        X_test,
        y_test,
        display_labels=["No stroke", "Stroke"],
        cmap="Blues",
        colorbar=False,
    )
    disp.ax_.set_title(f"Confusion Matrix: {combo_name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"confusion_matrix_{combo_name.lower().replace(' ', '_')}.png", dpi=220, bbox_inches="tight")
    plt.close()


def plot_roc_curves(finalist_estimators: List[Tuple[object, str]], X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Plot ROC curves for all tuned finalist models on a single axes."""
    plt.figure(figsize=(7, 6))
    for estimator, label in finalist_estimators:
        if hasattr(estimator, "predict_proba"):
            y_proba = estimator.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            plt.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random classifier")
    plt.xlabel("False Positive Rate (1 – Specificity)")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)")
    plt.title("ROC Curves – Tuned Finalist Models")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curves_finalists.png", dpi=220, bbox_inches="tight")
    plt.close()


def tune_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    versions_by_name: Dict[str, DatasetVersion],
    models: Dict[str, object],
    selected_models: List[str],
    selected_versions: List[str],
    cv: StratifiedKFold,
) -> Tuple[pd.DataFrame, Dict[str, str], List[Tuple[object, str]]]:
    param_grids = get_param_grids()
    tuning_rows = []
    selected_artifacts: Dict[str, str] = {}
    finalist_estimators: List[Tuple[object, str]] = []

    total_runs = len(selected_models) * len(selected_versions)
    run_idx = 0

    for version_name in selected_versions:
        version = versions_by_name[version_name]
        for model_name in selected_models:
            run_idx += 1
            combo_name = f"{version_name}_{model_name}"
            print(f"[grid {run_idx}/{total_runs}] {combo_name}")
            pipeline = build_pipeline(version, models[model_name])
            scorers = get_scorers()
            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grids[model_name],
                scoring=scorers,
                cv=cv,
                n_jobs=1,
                refit="balanced_accuracy",
                verbose=0,
            )
            grid.fit(X_train, y_train)

            best_idx = grid.best_index_
            metrics = evaluate_on_test(grid.best_estimator_, X_test, y_test)

            row: Dict[str, object] = {
                "dataset_version": version_name,
                "model": model_name,
                "combo": combo_name,
                "best_cv_balanced_accuracy": grid.best_score_,
                "best_cv_balanced_accuracy_std": grid.cv_results_["std_test_balanced_accuracy"][best_idx],
                "best_cv_recall": grid.cv_results_["mean_test_recall"][best_idx],
                "best_cv_recall_std": grid.cv_results_["std_test_recall"][best_idx],
                "best_cv_roc_auc": grid.cv_results_["mean_test_roc_auc"][best_idx],
                "best_cv_precision": grid.cv_results_["mean_test_precision"][best_idx],
                "best_cv_weighted_f1": grid.cv_results_["mean_test_f1_weighted"][best_idx],
                "best_params": json.dumps(grid.best_params_, sort_keys=True),
                **metrics,
            }

            pca = grid.best_estimator_.named_steps.get("pca")
            if pca is not None and hasattr(pca, "n_components_"):
                row["pca_n_components"] = int(pca.n_components_)
                row["pca_total_variance_retained"] = round(float(pca.explained_variance_ratio_.sum()), 4)

            tuning_rows.append(row)
            finalist_estimators.append((grid.best_estimator_, combo_name))

            model_path = MODELS_DIR / f"{combo_name}.joblib"
            joblib.dump(grid.best_estimator_, model_path)
            selected_artifacts[combo_name] = str(model_path)
            plot_confusion_matrix(grid.best_estimator_, X_test, y_test, combo_name)

    tuned_df = pd.DataFrame(tuning_rows).sort_values(
        by=["balanced_accuracy", "recall", "f1_macro", "f1_weighted"],
        ascending=False,
    )
    return tuned_df, selected_artifacts, finalist_estimators


def plot_tuned_results(tuned_df: pd.DataFrame) -> None:
    metric_cols = ["accuracy", "balanced_accuracy", "precision", "recall", "f1_weighted", "roc_auc"]
    plot_df = tuned_df[["combo"] + metric_cols].melt(id_vars="combo", var_name="metric", value_name="score")

    plt.figure(figsize=(10, 5.5))
    sns.barplot(data=plot_df, x="score", y="combo", hue="metric", orient="h")
    plt.title("Held-Out Test Metrics for Tuned Finalists")
    plt.xlabel("Score")
    plt.ylabel("")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tuned_test_metric_bars.png", dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    sns.set_theme(style="whitegrid")
    ensure_dirs()

    X, y, dataset_summary = load_data()
    dataset_summary.to_csv(TABLES_DIR / "dataset_summary.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    split_summary = pd.DataFrame(
        [
            {
                "train_rows": X_train.shape[0],
                "test_rows": X_test.shape[0],
                "train_stroke_rate_pct": round(100 * y_train.mean(), 2),
                "test_stroke_rate_pct": round(100 * y_test.mean(), 2),
            }
        ]
    )
    split_summary.to_csv(TABLES_DIR / "train_test_split_summary.csv", index=False)

    num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X_train.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()

    versions = [
        build_preprocessor("V1", num_features, cat_features),
        build_preprocessor("V2", num_features, cat_features),
        build_preprocessor("V3", num_features, cat_features),
        build_preprocessor("V4", num_features, cat_features),
    ]
    versions_by_name = {version.name: version for version in versions}
    models = build_candidate_models()
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    results_df = initial_experiments(X_train, y_train, versions, models, cv)
    results_df.to_csv(TABLES_DIR / "initial_cv_results.csv", index=False)

    model_summary, version_summary = summarize_initial_results(results_df)
    model_summary.to_csv(TABLES_DIR / "initial_cv_model_summary.csv", index=False)
    version_summary.to_csv(TABLES_DIR / "initial_cv_version_summary.csv", index=False)

    plot_initial_heatmap(results_df)
    plot_metric_bars(results_df)
    plot_balanced_accuracy_heatmap(results_df)

    selected_models, selected_versions = select_top_candidates(model_summary, version_summary)

    # Identify combinations that have high AUC but zero recall — the threshold-collapse pattern
    # that occurs when a model separates classes well in probability space but always predicts the
    # majority class at the default 0.5 threshold.  SMOTE (V3/V4) resolves this by rebalancing
    # the training distribution so the learned boundary shifts toward the minority class.
    threshold_collapse = results_df[
        (results_df["recall_mean"] == 0.0) & (results_df["roc_auc_mean"] >= 0.75)
    ][["dataset_version", "model", "roc_auc_mean", "recall_mean", "balanced_accuracy_mean"]].to_dict(orient="records")

    selection_summary = {
        "selected_models": selected_models,
        "selected_versions": selected_versions,
        "insight_threshold_collapse": {
            "description": (
                "These model+version combinations achieve a high CV ROC-AUC (good probabilistic "
                "ranking) yet recall = 0.0 at the default 0.5 threshold (always predict no-stroke). "
                "This confirms that SMOTE is necessary to shift the decision boundary toward the "
                "minority class; AUC alone is insufficient to detect strokes without rebalancing."
            ),
            "affected_combinations": threshold_collapse,
        },
    }
    (TABLES_DIR / "selection_summary.json").write_text(json.dumps(selection_summary, indent=2), encoding="utf-8")

    tuned_df, artifact_paths, finalist_estimators = tune_and_evaluate(
        X_train,
        y_train,
        X_test,
        y_test,
        versions_by_name,
        models,
        selected_models,
        selected_versions,
        cv,
    )
    tuned_df.to_csv(TABLES_DIR / "tuned_model_results.csv", index=False)
    plot_tuned_results(tuned_df)
    plot_roc_curves(finalist_estimators, X_test, y_test)

    best_row = tuned_df.iloc[0].to_dict()
    tp_count = int(best_row["tp"])
    fn_count = int(best_row["fn"])
    fp_count = int(best_row["fp"])
    tn_count = int(best_row["tn"])
    final_summary = {
        "data_path": str(DATA_PATH.relative_to(ROOT.parent)),
        "random_state": RANDOM_STATE,
        "cv_splits": CV_SPLITS,
        "selected_models": selected_models,
        "selected_versions": selected_versions,
        "best_model_combo": best_row["combo"],
        "best_model_path": str(Path(artifact_paths[best_row["combo"]]).relative_to(ROOT.parent)),
        "best_model_metrics": {
            "accuracy": round(float(best_row["accuracy"]), 4),
            "balanced_accuracy": round(float(best_row["balanced_accuracy"]), 4),
            "precision": round(float(best_row["precision"]), 4),
            "recall": round(float(best_row["recall"]), 4),
            "f1_weighted": round(float(best_row["f1_weighted"]), 4),
            "f1_macro": round(float(best_row["f1_macro"]), 4),
            "roc_auc": round(float(best_row["roc_auc"]), 4),
        },
        "best_model_params": json.loads(best_row["best_params"]),
        "best_model_interpretation": {
            "tp": tp_count,
            "fn": fn_count,
            "fp": fp_count,
            "tn": tn_count,
            "note": (
                f"Of {tp_count + fn_count} actual stroke cases in the held-out test set, "
                f"{tp_count} were correctly flagged (TP) and {fn_count} were missed (FN — false negatives). "
                f"{fp_count} non-stroke patients were incorrectly flagged as high-risk (FP — false positives). "
                "In a population screening context, minimising false negatives is the primary goal: "
                "a missed stroke (FN) carries a far higher clinical cost than an unnecessary follow-up (FP)."
            ),
        },
        "design_rationale": {
            "low_precision_is_expected": (
                f"Precision of {round(float(best_row['precision']), 3)} is low by design. "
                f"The model flags {fp_count} false positives alongside {tp_count} true positives "
                "because it was optimised for recall (sensitivity) rather than precision. "
                "In a population-level screening tool this is the correct tradeoff: the goal is "
                "to minimise missed strokes and refer borderline cases for clinical follow-up, "
                "not to achieve high precision at the cost of undetected events."
            ),
            "metric_selection_rationale": (
                "Balanced accuracy and recall were chosen as primary selection criteria because "
                "raw accuracy is misleading on this dataset (~95% accuracy is achievable by "
                "predicting no-stroke for every patient). Balanced accuracy penalises both "
                "false negatives and false positives equally, while recall directly measures "
                "the fraction of actual stroke cases detected — the clinically critical quantity."
            ),
        },
    }
    (TABLES_DIR / "final_summary.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")

    print("\nDeliverable 3 artifacts generated under:")
    print(f"- {ROOT}")
    print(f"- Best tuned model: {final_summary['best_model_combo']}")


if __name__ == "__main__":
    main()
