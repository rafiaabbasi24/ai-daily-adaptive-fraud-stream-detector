import streamlit as st
import pandas as pd
import numpy as np
import time
from typing import Tuple, List, Dict

# River (online ML) imports
from river import compose
from river import linear_model
from river import preprocessing
from river import metrics
from river import tree

# Scikit‑learn for offline evaluation utilities
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------


def load_data(file: bytes) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    file : bytes
        The uploaded file content.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the transaction data.
    """
    df = pd.read_csv(pd.io.common.BytesIO(file))
    # Ensure required columns exist
    required = {"amount", "type", "oldbalanceOrg", "newbalanceOrig",
                "oldbalanceDest", "newbalanceDest", "isFraud"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.stop()
    return df


def init_model() -> Tuple[compose.Pipeline, metrics.ROCAUC]:
    """
    Initialise the online learning pipeline and the evaluation metric.

    Returns
    -------
    model : river.compose.Pipeline
        The online model that will be updated incrementally.
    metric : river.metrics.ROCAUC
        Metric used to track model performance in real time.
    """
    # Simple preprocessing + logistic regression
    model = compose.Pipeline(
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression()
    )
    # Alternative: a Hoeffding Tree
    # model = compose.Pipeline(
    #     preprocessing.StandardScaler(),
    #     tree.HoeffdingTreeClassifier()
    # )
    metric = metrics.ROCAUC()
    return model, metric


def stream_process(
    df: pd.DataFrame,
    model: compose.Pipeline,
    metric: metrics.ROCAUC,
    progress_placeholder,
    result_placeholder,
    batch_size: int = 1,
    delay: float = 0.0,
) -> List[Dict]:
    """
    Process the DataFrame row‑by‑row, mimicking a live stream.

    Parameters
    ----------
    df : pd.DataFrame
        The transaction data.
    model : river.compose.Pipeline
        Online model that will be updated.
    metric : river.metrics.ROCAUC
        Metric tracking performance.
    progress_placeholder : streamlit.empty
        Placeholder to update progress bar.
    result_placeholder : streamlit.empty
        Placeholder to show the latest prediction.
    batch_size : int, optional
        Number of rows processed per UI update, by default 1.
    delay : float, optional
        Artificial delay (seconds) between batches to visualise streaming,
        by default 0.0.

    Returns
    -------
    List[Dict]
        List of dictionaries containing the original row, prediction and
        probability.
    """
    results = []
    total = len(df)
    progress_bar = progress_placeholder.progress(0)

    for i in range(0, total, batch_size):
        batch = df.iloc[i : i + batch_size]

        for _, row in batch.iterrows():
            # River expects a dict of features and a target label
            features = {
                "amount": row["amount"],
                "type": row["type"],
                "oldbalanceOrg": row["oldbalanceOrg"],
                "newbalanceOrig": row["newbalanceOrig"],
                "oldbalanceDest": row["oldbalanceDest"],
                "newbalanceDest": row["newbalanceDest"],
            }
            true_label = int(row["isFraud"])

            # Predict before learning
            y_pred = model.predict_one(features)
            y_proba = model.predict_proba_one(features).get(1, 0.0)

            # Update metric with the true label and predicted probability
            metric.update(true_label, y_proba)

            # Store result for later offline evaluation
            results.append(
                {
                    **features,
                    "true_label": true_label,
                    "pred_label": int(y_pred),
                    "pred_proba": y_proba,
                }
            )

            # Show the latest prediction in the UI
            result_placeholder.table(
                pd.DataFrame(
                    [
                        {
                            "Amount": row["amount"],
                            "Type": row["type"],
                            "Predicted Fraud": bool(y_pred),
                            "Probability": f"{y_proba:.4f}",
                            "Actual Fraud": bool(true_label),
                        }
                    ]
                )
            )

            # Simulate real‑time delay if requested
            if delay > 0:
                time.sleep(delay)

        # Update progress bar after each batch
        progress = min((i + batch_size) / total, 1.0)
        progress_bar.progress(progress)

    # Final progress update
    progress_bar.progress(1.0)
    return results


def compute_offline_metrics(results: List[Dict]) -> Tuple[str, np.ndarray]:
    """
    Compute classification report and confusion matrix from streamed results.

    Parameters
    ----------
    results : List[Dict]
        List returned by ``stream_process``.

    Returns
    -------
    report : str
        Textual classification report.
    cm : np.ndarray
        Confusion matrix (2x2).
    """
    y_true = [r["true_label"] for r in results]
    y_pred = [r["pred_label"] for r in results]

    report = classification_report(y_true, y_pred, digits=4, target_names=["Legit", "Fraud"])
    cm = confusion_matrix(y_true, y_pred)
    return report, cm


# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------


def main() -> None:
    """
    Entry point for the Streamlit application.
    """
    st.set_page_config(page_title="Real‑Time Adaptive Fraud Detector", layout="wide")
    st.title("🔍 Real‑Time Adaptive Fraud Detector for Transaction Streams")
    st.markdown(
        """
        This demo streams transaction data row‑by‑row, makes an online fraud prediction,
        updates the model continuously and displays live performance metrics.
        """
    )

    # Sidebar: file upload and configuration
    with st.sidebar:
        st.header("Configuration")
        uploaded_file = st.file_uploader(
            "Upload a CSV file with transaction data",
            type=["csv"],
            help=(
                "The CSV must contain the columns: "
                "`amount, type, oldbalanceOrg, newbalanceOrig, "
                "`oldbalanceDest, newbalanceDest, isFraud`."
            ),
        )
        batch_size = st.slider("Batch size (rows per UI update)", 1, 100, 10)
        delay = st.slider("Artificial delay (seconds)", 0.0, 2.0, 0.0, step=0.1)
        start_button = st.button("Start Streaming", disabled=uploaded_file is None)

    if not uploaded_file:
        st.info("Awaiting CSV upload...")
        st.stop()

    # Load data once (cached to avoid re‑reading on every rerun)
    @st.cache_data(show_spinner=False)
    def cached_load(file_bytes):
        return load_data(file_bytes)

    df = cached_load(uploaded_file.getvalue())

    # Initialise model and metric (cached to persist across runs)
    @st.cache_resource(show_spinner=False)
    def cached_init():
        return init_model()

    model, metric = cached_init()

    # Placeholders for dynamic UI components
    progress_placeholder = st.empty()
    result_placeholder = st.empty()
    metric_placeholder = st.empty()
    offline_report_placeholder = st.empty()
    confusion_placeholder = st.empty()

    if start_button:
        with st.spinner("Streaming data..."):
            results = stream_process(
                df,
                model,
                metric,
                progress_placeholder,
                result_placeholder,
                batch_size=batch_size,
                delay=delay,
            )

        # Show final online metric
        metric_placeholder.metric(
            label="Online ROC‑AUC (incremental)",
            value=f"{metric.get():.4f}",
        )

        # Compute and display offline evaluation
        report, cm = compute_offline_metrics(results)
        offline_report_placeholder.subheader("📊 Classification Report (offline)")
        offline_report_placeholder.text(report)

        confusion_placeholder.subheader("🧩 Confusion Matrix")
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Legit", "Actual Fraud"],
            columns=["Predicted Legit", "Predicted Fraud"],
        )
        confusion_placeholder.table(cm_df)

    else:
        st.info("Configure parameters and click **Start Streaming** to begin.")


if __name__ == "__main__":
    main()
