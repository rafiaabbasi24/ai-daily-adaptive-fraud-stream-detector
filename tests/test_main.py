"""tests/test_main.py
Unit tests for the core functionality of the Real-Time Adaptive Fraud Detector.

These tests cover:
- Data preprocessing utilities.
- Initial model training with a static batch.
- Online learning updates using River.
- End‑to‑end prediction consistency.

The test suite uses only public packages (pandas, numpy, scikit‑learn, river, pytest)
and assumes the project exposes an ``AdaptiveFraudDetector`` class in
``src/detector.py`` with the following public API:

    detector = AdaptiveFraudDetector()
    detector.fit(X_batch, y_batch)          # batch training
    prob = detector.predict_proba_one(row)  # returns probability of fraud
    detector.learn_one(row, label)          # online update

If the actual implementation differs, adjust the imports / attribute names
accordingly.
"""

from __future__ import annotations

import pathlib
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import the project code – adjust the path if the package layout changes.
# The tests assume the repository root is on the Python path (e.g. via
# ``pip install -e .`` or ``PYTHONPATH=.``
from src.detector import AdaptiveFraudDetector


def _generate_synthetic_data(
    n_samples: int = 1_000,
    n_features: int = 10,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Create a synthetic classification dataset mimicking transaction data.

    The function returns a DataFrame with feature columns named ``f0`` … ``f{n-1}``
    and a binary target series called ``is_fraud``.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.6),
        n_redundant=int(n_features * 0.2),
        n_repeated=0,
        n_classes=2,
        weights=[0.95, 0.05],  # fraud is rare
        flip_y=0.01,
        class_sep=1.0,
        random_state=random_state,
    )
    feature_names = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    # Add a timestamp column to emulate streaming data
    df["timestamp"] = pd.date_range(
        start="2022-01-01", periods=n_samples, freq="T"
    )
    # Shuffle rows to avoid any temporal leakage
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    target = pd.Series(y, name="is_fraud")
    return df, target


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Simple preprocessing used by the tests.

    - Drop the timestamp column.
    - Scale numeric features with StandardScaler.
    """
    df_clean = df.drop(columns=["timestamp"]).copy()
    scaler = StandardScaler()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])
    return df_clean


@pytest.fixture(scope="module")
def synthetic_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Module‑level fixture that provides a reproducible synthetic dataset."""
    df, y = _generate_synthetic_data()
    return df, y


def test_preprocess_removes_timestamp_and_scales(synthetic_dataset):
    """Verify that preprocessing removes the timestamp and scales numeric columns."""
    df, _ = synthetic_dataset
    processed = _preprocess(df)

    # Timestamp must be absent
    assert "timestamp" not in processed.columns

    # All numeric columns should have mean approx 0 and std approx 1
    numeric = processed.select_dtypes(include=[np.number])
    means = numeric.mean().abs()
    stds = (numeric.std(ddof=0) - 1).abs()
    assert (means < 1e-6).all(), "Features are not centered around 0"
    assert (stds < 1e-6).all(), "Features are not scaled to unit variance"


def test_batch_training_and_initial_prediction(synthetic_dataset):
    """Train on a static batch and assert that predictions are probabilities
    between 0 and 1 and that the model distinguishes fraud from legit."""
    df, y = synthetic_dataset
    X = _preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
    )

    detector = AdaptiveFraudDetector()
    detector.fit(X_train, y_train)

    # Predict on the hold‑out set
    probs = X_test.apply(detector.predict_proba_one, axis=1)
    assert probs.between(0.0, 1.0).all(), "Probabilities must be within [0, 1]"

    # Simple sanity check: average fraud probability should be higher for
    # the true fraud class.
    fraud_mask = y_test == 1
    avg_fraud = probs[fraud_mask].mean()
    avg_legit = probs[~fraud_mask].mean()
    assert avg_fraud > avg_legit, "Model should assign higher scores to fraud cases"


def test_online_learning_improves_detection(synthetic_dataset):
    """Simulate a stream of transactions and verify that online updates
    improve the model's ability to detect fraud."""
    df, y = synthetic_dataset
    X = _preprocess(df)

    # Split the data: first 70 % for initial batch training, rest for streaming.
    split_idx = int(0.7 * len(df))
    X_initial, X_stream = X.iloc[:split_idx], X.iloc[split_idx:]
    y_initial, y_stream = y.iloc[:split_idx], y.iloc[split_idx:]

    detector = AdaptiveFraudDetector()
    detector.fit(X_initial, y_initial)

    # Record baseline AUC on the streaming portion before any online updates.
    from sklearn.metrics import roc_auc_score

    baseline_probs = X_stream.apply(detector.predict_proba_one, axis=1)
    baseline_auc = roc_auc_score(y_stream, baseline_probs)

    # Perform online learning: for each new sample, predict then learn.
    updated_probs = []
    for _, row in X_stream.iterrows():
        prob = detector.predict_proba_one(row)
        updated_probs.append(prob)
        # Retrieve the true label (aligned by index)
        true_label = y_stream.loc[row.name]
        detector.learn_one(row, true_label)

    post_update_auc = roc_auc_score(y_stream, updated_probs)

    # The AUC after online updates should be at least as good as the baseline.
    # Small fluctuations are possible; we allow a tiny tolerance.
    assert post_update_auc >= baseline_auc - 0.01, (
        f"Online learning degraded performance: baseline {baseline_auc:.3f}, "
        f"post‑update {post_update_auc:.3f}"
    )


def test_predict_proba_one_returns_float_and_is_repeatable():
    """Check that ``predict_proba_one`` returns a float and that repeated calls
    without learning produce the same output (deterministic inference)."""
    # Create a tiny deterministic dataset
    data = {
        "f0": [0.5],
        "f1": [-1.2],
        "f2": [3.3],
    }
    df = pd.DataFrame(data)
    detector = AdaptiveFraudDetector()
    # Fit on a single example with label 0 to initialise internal models
    detector.fit(df, pd.Series([0]))

    row = df.iloc[0]
    prob_first = detector.predict_proba_one(row)
    prob_second = detector.predict_proba_one(row)

    assert isinstance(prob_first, float), "Probability must be a float"
    assert prob_first == prob_second, "Inference should be deterministic"


def test_detector_serialization(tmp_path: pathlib.Path, synthetic_dataset):
    """Validate that the detector can be saved to disk and re‑loaded without loss."""
    df, y = synthetic_dataset
    X = _preprocess(df)

    detector = AdaptiveFraudDetector()
    detector.fit(X, y)

    # Serialize
    model_path = tmp_path / "detector.pkl"
    detector.save(str(model_path))

    # Load a new instance
    new_detector = AdaptiveFraudDetector.load(str(model_path))

    # Predictions from both instances should match on a random sample
    sample_idx = random.randint(0, len(X) - 1)
    row = X.iloc[sample_idx]
    prob_original = detector.predict_proba_one(row)
    prob_loaded = new_detector.predict_proba_one(row)

    assert np.isclose(prob_original, prob_loaded), "Loaded model predictions differ"


# End of test suite.