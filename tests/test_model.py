import pytest
import pandas as pd
import numpy as np

# The project is expected to expose an adaptive fraud detector.
# It may live in `model.py` at the repository root or inside a `src` package.
# We try both import locations for flexibility.
try:
    from model import AdaptiveFraudDetector  # type: ignore
except ImportError:  # pragma: no cover
    try:
        from src.model import AdaptiveFraudDetector  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Could not import AdaptiveFraudDetector. Ensure the project defines "
            "`AdaptiveFraudDetector` in either `model.py` or `src/model.py`."
        ) from exc


def _create_synthetic_batch(seed: int = 0) -> pd.DataFrame:
    """Create a small synthetic transaction batch.

    The batch contains a mix of fraudulent and legitimate transactions.
    Features are deliberately simple to keep the test fast and deterministic.
    """
    rng = np.random.default_rng(seed)
    n = 10
    df = pd.DataFrame(
        {
            "amount": rng.integers(1, 5000, size=n).astype(float),
            "hour": rng.integers(0, 24, size=n).astype(int),
            "country": rng.choice(["US", "GB", "FR", "DE"], size=n),
            "merchant_category": rng.choice(
                ["electronics", "groceries", "travel", "entertainment"], size=n
            ),
            # Binary label: 1 = fraud, 0 = legitimate
            "fraud": rng.choice([0, 1], size=n, p=[0.8, 0.2]),
        }
    )
    return df


@pytest.fixture
def detector():
    """Create a fresh AdaptiveFraudDetector for each test."""
    # The detector is expected to accept optional hyper‑parameters.
    # Using defaults keeps the test focused on behaviour rather than tuning.
    return AdaptiveFraudDetector()


def _fit_or_update(detector, df: pd.DataFrame):
    """Utility that calls the appropriate incremental learning method.

    The project may expose either a `fit`, `partial_fit`, or `update` method
    that accepts a pandas DataFrame. This helper abstracts that difference.
    """
    if hasattr(detector, "fit"):
        # Some implementations treat `fit` as incremental when a DataFrame is passed.
        detector.fit(df)
    elif hasattr(detector, "partial_fit"):
        detector.partial_fit(df)
    elif hasattr(detector, "update"):
        detector.update(df)
    else:
        # Fall back to learning one row at a time using River‑style API.
        if hasattr(detector, "learn_one"):
            for _, row in df.iterrows():
                x = row.drop("fraud").to_dict()
                y = row["fraud"]
                detector.learn_one(x, y)
        else:
            raise AttributeError(
                "Detector does not expose a known incremental learning interface."
            )


def _predict_proba(detector, X: pd.DataFrame) -> np.ndarray:
    """Return probability of the positive class for each row in ``X``."""
    if hasattr(detector, "predict_proba"):
        # Assume the method returns a 2‑column array like scikit‑learn.
        proba = detector.predict_proba(X)[:, 1]
    elif hasattr(detector, "predict_proba_one"):
        # River style – iterate over rows.
        proba = np.array([detector.predict_proba_one(row.to_dict())[1] for _, row in X.iterrows()])
    else:
        raise AttributeError("Detector lacks a `predict_proba` method.")
    return proba


def test_initial_fit_and_prediction(detector):
    """The detector should produce probabilities after an initial fit."""
    batch = _create_synthetic_batch(seed=42)
    _fit_or_update(detector, batch)

    # Predict on a fresh set of transactions (no labels needed).
    test_batch = _create_synthetic_batch(seed=99).drop(columns=["fraud"])
    probs = _predict_proba(detector, test_batch)

    # Probabilities must be numeric, finite and lie in [0, 1].
    assert probs.shape == (len(test_batch),)
    assert np.all(np.isfinite(probs)), "Probabilities contain NaN or infinite values."
    assert np.all((0.0 <= probs) & (probs <= 1.0)), "Probabilities fall outside [0, 1]."


def test_incremental_update_improves_detection(detector):
    """After seeing a clear fraudulent example, the model should assign higher fraud
    probability to similar transactions.
    """
    # 1️⃣ Initial training on a balanced batch without obvious patterns.
    init_batch = _create_synthetic_batch(seed=1)
    _fit_or_update(detector, init_batch)

    # 2️⃣ Create a transaction that is highly indicative of fraud.
    fraud_example = pd.DataFrame(
        {
            "amount": [4500.0],          # unusually high amount
            "hour": [2],                 # odd hour
            "country": ["GB"],
            "merchant_category": ["electronics"],
            "fraud": [1],
        }
    )
    # Record the model's probability before seeing the example.
    pre_probs = _predict_proba(detector, fraud_example.drop(columns=["fraud"]))

    # 3️⃣ Incrementally update the model with the fraudulent example.
    _fit_or_update(detector, fraud_example)

    # 4️⃣ Predict again on the same transaction.
    post_probs = _predict_proba(detector, fraud_example.drop(columns=["fraud"]))

    # The probability of fraud should increase after the model has observed a similar case.
    assert post_probs[0] > pre_probs[0], (
        f"Model probability did not increase after learning a fraud example: "
        f"pre={pre_probs[0]:.4f}, post={post_probs[0]:.4f}"
    )
    # Ensure the probability is still a valid probability.
    assert 0.0 <= post_probs[0] <= 1.0


def test_handling_of_unseen_categories(detector):
    """The model should not crash when encountering categorical values it has not seen."""
    # Train on a batch that only contains a subset of possible countries.
    train_batch = pd.DataFrame(
        {
            "amount": [100.0, 200.0],
            "hour": [12, 13],
            "country": ["US", "US"],
            "merchant_category": ["groceries", "groceries"],
            "fraud": [0, 0],
        }
    )
    _fit_or_update(detector, train_batch)

    # Create a test row with a new country and merchant category.
    unseen_batch = pd.DataFrame(
        {
            "amount": [1500.0],
            "hour": [3],
            "country": ["JP"],  # not present in training data
            "merchant_category": ["luxury"],  # not present in training data
        }
    )
    # The prediction should succeed and return a probability in [0, 1].
    probs = _predict_proba(detector, unseen_batch)
    assert probs.shape == (1,)
    assert 0.0 <= probs[0] <= 1.0, "Probability for unseen categories is out of bounds."


def test_batch_update_consistency(detector):
    """Updating the model with a batch should be equivalent to updating row‑by‑row."""
    batch = _create_synthetic_batch(seed=7)

    # Clone the detector to have two independent instances.
    detector_batch = AdaptiveFraudDetector()
    detector_rowwise = AdaptiveFraudDetector()

    # Batch update.
    _fit_or_update(detector_batch, batch)

    # Row‑wise update using River‑style `learn_one` if available.
    if hasattr(detector_rowwise, "learn_one"):
        for _, row in batch.iterrows():
            x = row.drop("fraud").to_dict()
            y = row["fraud"]
            detector_rowwise.learn_one(x, y)
    else:
        # Fall back to the generic helper.
        _fit_or_update(detector_rowwise, batch)

    # Compare predictions on a new random set.
    test_set = _create_synthetic_batch(seed=21).drop(columns=["fraud"])
    probs_batch = _predict_proba(detector_batch, test_set)
    probs_rowwise = _predict_proba(detector_rowwise, test_set)

    # The two sets of probabilities should be almost identical (tolerance for floating‑point noise).
    np.testing.assert_allclose(
        probs_batch,
        probs_rowwise,
        rtol=1e-5,
        atol=1e-8,
        err_msg="Batch and row‑wise updates yielded different predictions.",
    )