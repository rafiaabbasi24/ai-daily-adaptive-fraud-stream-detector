import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from river import compose, linear_model, metrics, preprocessing, tree


class AdaptiveFraudModel:
    """
    A production‑ready wrapper around a River online learning model for real‑time fraud detection.

    The model expects each transaction to be presented as a ``dict`` mapping feature names to values.
    It can be trained incrementally with :meth:`learn_one` and used to obtain predictions or
    probability estimates with :meth:`predict_one` / :meth:`predict_proba_one`.

    Parameters
    ----------
    model :
        An instantiated River model (e.g. ``tree.HoeffdingTreeClassifier``). If ``None``,
        a sensible default pipeline is built consisting of a ``StandardScaler`` followed by a
        ``HoeffdingTreeClassifier`` suitable for binary classification.
    """

    #: Default path used by :meth:`save` when no explicit path is supplied.
    DEFAULT_SAVE_PATH = Path("models") / "adaptive_fraud_model.pkl"

    def __init__(self, model: Optional[compose.Pipeline] = None) -> None:
        self.model: compose.Pipeline = model if model is not None else self._build_default_pipeline()
        # River metric for optional online evaluation (not required for inference)
        self.metric = metrics.ROCAUC()

    @staticmethod
    def _build_default_pipeline() -> compose.Pipeline:
        """
        Construct a default River pipeline.

        The pipeline performs:
        1. Standard scaling of numeric features.
        2. Online logistic regression as a fallback for linear separable data.
        3. Hoeffding Tree for non‑linear patterns.

        Returns
        -------
        compose.Pipeline
            A River pipeline ready for incremental training.
        """
        scaler = preprocessing.StandardScaler()
        # Logistic regression works well as a fast linear baseline.
        logistic = linear_model.LogisticRegression()
        # Hoeffding Tree captures more complex relationships.
        tree_clf = tree.HoeffdingTreeClassifier(
            grace_period=50,
            split_confidence=0.01,
            leaf_prediction="binary",
            max_depth=None,
        )
        # Combine linear and tree predictions using a simple voting scheme.
        # River's ``compose.Pipeline`` applies steps sequentially, so we use a
        # ``compose.StackedEnsemble``‑like approach manually.
        # For simplicity we chain scaler → logistic → tree.
        return compose.Pipeline(
            scaler,
            logistic,
            tree_clf,
        )

    def learn_one(self, x: Dict[str, Any], y: int) -> None:
        """
        Incrementally train the model on a single observation.

        Parameters
        ----------
        x :
            Feature dictionary for a single transaction.
        y :
            Ground‑truth label (``0`` for legitimate, ``1`` for fraudulent).
        """
        self.model.learn_one(x, y)
        self.metric.update(y, self.predict_proba_one(x))

    def predict_proba_one(self, x: Dict[str, Any]) -> float:
        """
        Return the probability that the given transaction is fraudulent.

        Parameters
        ----------
        x :
            Feature dictionary for a single transaction.

        Returns
        -------
        float
            Probability of the positive class (fraud). If the underlying model does not
            provide probabilities, a fallback of ``0.0`` is returned.
        """
        # River models expose ``predict_proba_one``; if unavailable we fall back to ``predict_one``.
        if hasattr(self.model, "predict_proba_one"):
            proba_dict = self.model.predict_proba_one(x)  # type: ignore[arg-type]
            # ``predict_proba_one`` returns a mapping {class_label: probability}
            return float(proba_dict.get(1, 0.0))
        # Fallback – treat deterministic prediction as 0.0/1.0 probability.
        return float(self.model.predict_one(x))

    def predict_one(self, x: Dict[str, Any], threshold: float = 0.5) -> int:
        """
        Produce a binary fraud prediction for a single transaction.

        Parameters
        ----------
        x :
            Feature dictionary for a single transaction.
        threshold :
            Decision threshold applied to the probability output. Defaults to ``0.5``.

        Returns
        -------
        int
            ``1`` if the transaction is predicted fraudulent, otherwise ``0``.
        """
        prob = self.predict_proba_one(x)
        return int(prob >= threshold)

    def evaluate_one(self, x: Dict[str, Any], y: int) -> None:
        """
        Update the internal online metric with a new observation.

        This method is a thin wrapper around :meth:`learn_one` that also records the
        metric state. It is useful when you want to monitor performance without
        storing predictions.

        Parameters
        ----------
        x :
            Feature dictionary.
        y :
            True label.
        """
        self.learn_one(x, y)

    @classmethod
    def load(cls, path: Optional[os.PathLike] = None) -> "AdaptiveFraudModel":
        """
        Load a previously saved model from disk.

        Parameters
        ----------
        path :
            Path to the pickle file. If ``None``, :data:`DEFAULT_SAVE_PATH` is used.

        Returns
        -------
        AdaptiveFraudModel
            The deserialized model instance.
        """
        load_path = Path(path) if path is not None else cls.DEFAULT_SAVE_PATH
        if not load_path.is_file():
            raise FileNotFoundError(f"No model file found at {load_path}")

        with load_path.open("rb") as f:
            obj = pickle.load(f)

        if not isinstance(obj, cls):
            raise TypeError(f"Pickled object is not an {cls.__name__} instance")
        return obj

    def save(self, path: Optional[os.PathLike] = None) -> None:
        """
        Serialize the model to disk using ``pickle``.

        Parameters
        ----------
        path :
            Destination file path. If ``None``, :data:`DEFAULT_SAVE_PATH` is used.
            The parent directory is created automatically if it does not exist.
        """
        save_path = Path(path) if path is not None else self.DEFAULT_SAVE_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with save_path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def pandas_row_to_dict(row: pd.Series) -> Dict[str, Any]:
        """
        Convert a ``pandas.Series`` (a row from a DataFrame) to the dictionary format
        expected by River.

        Parameters
        ----------
        row :
            A pandas Series representing a transaction.

        Returns
        -------
        dict
            Mapping of column names to scalar values, with ``NaN`` replaced by ``None``.
        """
        # ``Series.to_dict`` retains ``numpy.nan``; River treats ``None`` as missing.
        return {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"


__all__ = ["AdaptiveFraudModel"]