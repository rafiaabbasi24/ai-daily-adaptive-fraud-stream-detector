import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Iterable, List

# River (online machine learning) imports
from river import compose
from river import preprocessing
from river import linear_model
from river import tree
from river import metrics

# Scikit‑learn for an initial batch training (optional)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# ----------------------------------------------------------------------
# Configuration constants
# ----------------------------------------------------------------------
DEFAULT_DATA_PATH = Path("data/transactions_sample.csv")
CHUNK_SIZE = 1000  # rows per chunk when streaming from a CSV file
TARGET_COLUMN = "is_fraud"  # binary target column name
CATEGORICAL_FEATURES = ["merchant", "category", "city"]
NUMERIC_FEATURES = ["amount", "hour_of_day", "day_of_week"]


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def load_initial_dataset(path: Path) -> pd.DataFrame:
    """
    Load a static dataset used for an optional initial batch training.
    The function expects the CSV to contain the columns listed in
    ``CATEGORICAL_FEATURES`` and ``NUMERIC_FEATURES`` plus the target column.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    required = set(CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TARGET_COLUMN])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")
    return df


def preprocess_batch(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """
    Perform a one‑hot encoding for categorical variables and return the
    transformed features, target series, and the fitted transformer.
    This is only used for the optional initial batch training.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    transformer = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    X_transformed = transformer.fit_transform(X)
    return pd.DataFrame(
        X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed,
        columns=transformer.get_feature_names_out(),
    ), y, transformer


def river_feature_extractor() -> compose.Transformer:
    """
    Build a River transformer that mirrors the scikit‑learn preprocessing
    used for the initial batch. It performs one‑hot encoding for categorical
    fields and standard scaling for numeric fields.
    """
    return compose.GenericTransformer(
        func=lambda x: {
            **{f: x[f] for f in NUMERIC_FEATURES},
            **{
                f + "_" + str(v): 1
                for f in CATEGORICAL_FEATURES
                for v in [x[f]]
                if pd.notna(v)
            },
        }
    )


def build_online_model() -> Tuple[compose.Pipeline, metrics.ClassificationReport]:
    """
    Construct the online learning pipeline and a classification report metric.
    The pipeline consists of:
        - feature extraction (one‑hot + numeric)
        - optional standardisation
        - a Hoeffding Tree classifier (good for streaming data)
    """
    model = compose.Pipeline(
        river_feature_extractor(),
        preprocessing.StandardScaler(),
        tree.HoeffdingTreeClassifier(
            grace_period=200,
            split_confidence=0.01,
            leaf_prediction="nba",  # naive Bayes at leaves
        ),
    )
    metric = metrics.ClassificationReport()
    return model, metric


def stream_csv_rows(path: Path, chunk_size: int = CHUNK_SIZE) -> Iterable[pd.Series]:
    """
    Lazily read a CSV file in chunks and yield one row (as a pandas Series)
    at a time. This mimics a real‑time data stream.
    """
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        for _, row in chunk.iterrows():
            yield row


def row_to_river_dict(row: pd.Series) -> Tuple[Dict[str, Any], int]:
    """
    Convert a pandas Series representing a transaction into a River
    feature dictionary and a binary label.
    """
    features = {
        **{f: row[f] for f in NUMERIC_FEATURES},
        **{
            f + "_" + str(row[f]): 1
            for f in CATEGORICAL_FEATURES
            if pd.notna(row[f])
        },
    }
    label = int(row[TARGET_COLUMN])
    return features, label


# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------
def main() -> None:
    """
    Entry point for the Streamlit application. Users can upload a CSV file
    containing transaction records, view a preview, and start a real‑time
    fraud detection simulation. Predictions and model performance metrics
    are displayed live.
    """
    st.set_page_config(page_title="Real‑Time Adaptive Fraud Detector", layout="wide")
    st.title("🔍 Real‑Time Adaptive Fraud Detector for Transaction Streams")

    # ------------------------------------------------------------------
    # Sidebar – configuration
    # ------------------------------------------------------------------
    st.sidebar.header("Configuration")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a CSV file with transactions",
        type=["csv"],
        help="The file must contain columns: "
        + ", ".join(CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TARGET_COLUMN]),
    )
    start_button = st.sidebar.button("Start Streaming Detection", type="primary")
    reset_button = st.sidebar.button("Reset Model")

    # ------------------------------------------------------------------
    # Session state handling
    # ------------------------------------------------------------------
    if "model" not in st.session_state or reset_button:
        st.session_state.model, st.session_state.metric = build_online_model()
        st.session_state.predictions = []  # type: List[Dict[str, Any]]
        st.session_state.processed = 0

    model: compose.Pipeline = st.session_state.model
    metric: metrics.ClassificationReport = st.session_state.metric

    # ------------------------------------------------------------------
    # Data preview
    # ------------------------------------------------------------------
    if uploaded_file is not None:
        try:
            df_preview = pd.read_csv(uploaded_file, nrows=5)
            st.subheader("Data preview (first 5 rows)")
            st.dataframe(df_preview)
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
            return
    else:
        # Show default sample if no file uploaded
        if DEFAULT_DATA_PATH.is_file():
            st.info(f"No file uploaded – using sample dataset at `{DEFAULT_DATA_PATH}`")
            df_preview = pd.read_csv(DEFAULT_DATA_PATH, nrows=5)
            st.subheader("Sample data preview")
            st.dataframe(df_preview)
        else:
            st.warning("Please upload a CSV file to begin.")
            return

    # ------------------------------------------------------------------
    # Streaming logic
    # ------------------------------------------------------------------
    if start_button:
        # Determine which path to stream
        stream_path = (
            Path(uploaded_file.name)
            if uploaded_file
            else DEFAULT_DATA_PATH
        )
        # If the uploaded file is an in‑memory BytesIO, we need to write it temporarily
        if uploaded_file and not stream_path.is_file():
            temp_path = Path("temp_uploaded.csv")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            stream_path = temp_path

        progress_placeholder = st.empty()
        table_placeholder = st.empty()
        metric_placeholder = st.empty()

        # Stream rows one by one
        for i, row in enumerate(stream_csv_rows(stream_path)):
            features, label = row_to_river_dict(row)

            # Predict before learning (online learning paradigm)
            y_pred = model.predict_one(features)
            y_pred_label = int(y_pred) if y_pred is not None else 0

            # Update model with the true label
            model.learn_one(features, label)

            # Update metric
            metric.update(label, y_pred_label)

            # Store prediction for UI display
            st.session_state.predictions.append(
                {
                    "index": i,
                    "transaction_id": row.get("transaction_id", i),
                    "predicted_fraud": y_pred_label,
                    "actual_fraud": label,
                    "amount": row.get("amount"),
                }
            )
            st.session_state.processed = i + 1

            # Refresh UI every 50 rows to avoid excessive re‑rendering
            if (i + 1) % 50 == 0 or (i + 1) == 1:
                progress_placeholder.progress(min(1.0, (i + 1) / 1000))
                df_display = pd.DataFrame(st.session_state.predictions[-10:])[
                    ["index", "transaction_id", "amount", "predicted_fraud", "actual_fraud"]
                ]
                table_placeholder.table(df_display)

                metric_placeholder.metric(
                    "Classification report (updated)",
                    metric.get(),
                )

        progress_placeholder.progress(1.0)
        st.success(f"Streaming completed – processed {st.session_state.processed} rows.")
        metric_placeholder.metric("Final classification report", metric.get())

        # Clean up temporary file if created
        if uploaded_file and stream_path.name == "temp_uploaded.csv":
            try:
                stream_path.unlink()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    st.caption(
        "© 2026 Real‑Time Adaptive Fraud Detector. "
        "Built with Streamlit, River, pandas, and scikit‑learn."
    )


if __name__ == "__main__":
    main()
