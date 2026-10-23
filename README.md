# Real-Time Adaptive Fraud Detector for Transaction Streams

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture & Workflow](#architecture--workflow)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Data Requirements](#data-requirements)
- [Model Details](#model-details)
- [Evaluation & Monitoring](#evaluation--monitoring)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview
The **Real-Time Adaptive Fraud Detector** is a streaming machine‑learning system that ingests transaction data, detects fraudulent activity on the fly, and continuously adapts to evolving fraud patterns. It combines batch preprocessing with online learning (via the **River** library) and provides an interactive dashboard built with **Streamlit** for monitoring model performance, visualizing alerts, and tweaking parameters in real time.

---

## Features
- **Real‑time inference** on incoming transaction streams.
- **Online learning**: the model updates incrementally without retraining from scratch.
- **Drift detection** using River’s built‑in change‑detection methods.
- **Interactive dashboard** for visualizing transaction flow, fraud alerts, and model metrics.
- **Modular pipeline**: easy to swap data sources, feature engineering steps, or classifiers.
- **Production‑ready**: containerizable, configurable via environment variables, and includes logging.

---

## Architecture & Workflow
```
+-------------------+      +-------------------+      +-------------------+
| Transaction Source| ---> | Stream Processor  | ---> | Online Model (River)|
| (Kafka / CSV / API) |   | (pandas + River) |   |   + Drift Detector |
+-------------------+      +-------------------+      +-------------------+
                                 |
                                 v
                         +-------------------+
                         | Streamlit Dashboard|
                         +-------------------+
```

1. **Ingestion** – Transactions are read from a configurable source (Kafka topic, CSV file, or REST API) and converted into a Pandas DataFrame.
2. **Pre‑processing** – Feature engineering (e.g., time‑based aggregations, categorical encoding) is applied using Pandas.
3. **Online Prediction** – The River model predicts fraud probability for each transaction.
4. **Model Update** – True labels (when available) are streamed back to update the model incrementally.
5. **Drift Detection** – A change detector flags significant performance shifts, prompting optional alerts.
6. **Dashboard** – Streamlit visualizes live metrics, recent alerts, and allows on‑the‑fly parameter tuning.

---

## Tech Stack
| Layer | Library / Tool |
|-------|----------------|
| Language | Python 3.10+ |
| Data handling | pandas |
| Batch & online ML | scikit-learn, river |
| Visualization & UI | streamlit |
| Messaging (optional) | kafka-python |
| Logging | loguru |
| Environment management | pip, virtualenv / conda |
| Containerization | Docker (optional) |

---

## Installation

### Prerequisites
- Python 3.10 or newer
- `git` command line tool
- (Optional) Docker if you prefer container deployment

### Steps
```bash
# 1. Clone the repository
git clone https://github.com/your-org/real-time-fraud-detector.git
cd real-time-fraud-detector

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# 3. Install required packages
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Optional) Install development tools
pip install -r requirements-dev.txt
```

### Docker (optional)
```bash
docker build -t fraud-detector:latest .
docker run -p 8501:8501 fraud-detector:latest
```

---

## Configuration
All configurable parameters are stored in `config.yaml`. Key sections include:

```yaml
data:
  source: "csv"                # csv | kafka | api
  csv_path: "data/transactions.csv"
  kafka:
    bootstrap_servers: "localhost:9092"
    topic: "transactions"

model:
  type: "HoeffdingTreeClassifier"
  drift_detector: "ADWIN"
  learning_rate: 0.01

dashboard:
  port: 8501
  refresh_interval: 2          # seconds
```

You can override any setting with environment variables prefixed by `FRAUD_`, e.g.:

```bash
export FRAUD_DATA_SOURCE=kafka
export FRAUD_KAFKA_BOOTSTRAP_SERVERS=broker:9092
```

---

## Running the Application

### 1. Start the Stream Processor
```bash
python src/processor.py --config config.yaml
```
The processor will continuously read transactions, predict fraud, and update the model.

### 2. Launch the Streamlit Dashboard
```bash
streamlit run src/dashboard.py --server.port 8501
```
Open your browser at `http://localhost:8501` to explore live metrics.

### 3. End‑to‑End Demo (CSV source)
```bash
# In one terminal
python src/processor.py --config config.yaml

# In another terminal
streamlit run src/dashboard.py
```
The demo reads `data/transactions.csv`, streams rows with a 0.5 s pause, and displays predictions in real time.

---

## Data Requirements
The system expects a tabular transaction record with at least the following columns:

| Column | Description |
|--------|-------------|
| `transaction_id` | Unique identifier |
| `timestamp` | ISO‑8601 datetime string |
| `amount` | Transaction amount (float) |
| `merchant_id` | Categorical merchant identifier |
| `customer_id` | Categorical customer identifier |
| `device_id` | Device identifier (optional) |
| `is_fraud` | Ground‑truth label (0/1). Required for model updates; can be `null` for unlabeled streams. |

Additional features can be added; just update `src/features.py` accordingly.

---

## Model Details
- **Base classifier**: `river.tree.HoeffdingTreeClassifier` (default). Can be swapped for any River estimator (e.g., `river.linear_model.LogisticRegression`).
- **Drift detection**: `river.drift.ADWIN` monitors the running loss; when a drift is detected, a warning is logged and the dashboard highlights the event.
- **Feature pipeline**: Implemented in `src/features.py`. Uses `river.compose` for online encoders (e.g., `OneHotEncoder`, `StandardScaler`).

**Training / Update Loop (simplified):**
```python
for tx in stream:
    X = feature_extractor.transform_one(tx)
    y_pred = model.predict_one(X)
    # Log prediction, send to dashboard
    if tx.get("is_fraud") is not None:
        model.learn_one(X, tx["is_fraud"])
        drift_detector.update(model.loss_one(y_pred, tx["is_fraud"]))
```

All model state is persisted to `models/model.pkl` every 5 minutes (or on graceful shutdown) using River’s `pickle` utilities.

---

## Evaluation & Monitoring
- **Metrics displayed on the dashboard**: ROC‑AUC, Precision, Recall, F1‑Score, and a rolling confusion matrix.
- **Drift alerts**: Shown as red banners with timestamp and detector statistics.
- **Logs**: Written to `logs/app.log` with rotation (10 MB per file, keep 5 files) using `loguru`.

You can export the current model for offline analysis:
```bash
python src/export_model.py --output models/offline.pkl
```

---

## Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/awesome-feature`).
3. Install development dependencies (`pip install -r requirements-dev.txt`).
4. Run tests: `pytest -vv`.
5. Ensure code passes `flake8` and `black` formatting.
6. Submit a Pull Request with a clear description and reference to any related issue.

Please see `CONTRIBUTING.md` for detailed guidelines.

---

## License
This project is licensed under the **MIT License** – see the `LICENSE` file for details.