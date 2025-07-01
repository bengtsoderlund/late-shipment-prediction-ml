# Predicting Late Shipments: An End-to-End Machine Learning Project

## Project Overview

Timely delivery is a critical factor in supply chain management, as late shipments can lead to customer dissatisfaction, revenue loss, and increased operational costs. In this project, I use machine learning to develop predictive models for a global sports and outdoor equipment retailer to proactively identify high-risk shipments before delays occur.

The project follows modern best practices in data science and machine learning development:

- **Modular pipeline**: Cleanly separated stages for loading, cleaning, feature engineering, preprocessing, training, and evaluation.
- **MLflow integration**: Tracks experiments and hyperparameter tuning for reproducibility.
- **REST API deployment**: Trained models are served using **FastAPI**, packaged in a **Docker** container, and deployed via **Render**.
- **Automated testing**: FastAPI routes are verified using **pytest**, including landing page, health check, and prediction endpoints.
- **Structured logging**: Unified logging system records progress and errors for easier debugging and traceability.
- **Exploratory data analysis (EDA)**: Initial insights and feature selection decisions are documented in a dedicated Jupyter notebook.

**Tech Stack Overview:**

> `Python 3.11`, `pandas`, `NumPy`, `scikit-learn`, `FastAPI`, `Pydantic`, `MLflow`, `Docker`, `Render`, `Uvicorn`, `pytest`, `joblib`, `RobustScaler`, `OneHotEncoder`, `OrdinalEncoder`, `logging`, `pathlib`, `datetime`.

This combination ensures a scalable, production-ready workflow aligned with modern MLOps and full-stack data science standards.

## Dataset Information

The data for this project is provided by DataCo and is publicly available on Kaggle ([link](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)). It contains approximately 180,000 transactions that are shipped to 164 unique countries over the span of three years. The dataset provides a comprehensive view of supply chain operations, including:
- **Order details:** Order IDs, order destination, and order dates.
- **Product information:** Product IDs, pricing, and sales.
- **Customer data:** Customer segment and locations.
- **Shipping records:** Scheduled vs. actual delivery times, shipping mode, and delivery status.

## Business Problem

The dataset reveals that 57% of completed shipments are late by at least one day, and more than 7% are delayed by three or more days. To improve on-time delivery rates, businesses need to identify high-risk shipments early and take proactive measures to mitigate delays.

## Solution Approach

To address the challenge of shipment delays, this project builds machine learning models that classify whether a shipment will arrive late or very late. These predictive tools help prioritize shipments that require early intervention or adjusted logistics handling.

Two Random Forest classifiers were trained:
1. **Late Order Model (optimized for accuracy)**  
   Predicts whether an order will be delivered late (by at least one day).

2. **Very Late Order Model (optimized for recall)**  
   Predicts whether an order will be three or more days late prioritizing recall to flag high-risk shipments early in the process.

## Model Performance

Two Random Forest models were developed and evaluated on a hold-out test set (25% of the dataset), with each optimized for a metric aligned to its business use case.

- **Late Shipment Model (optimized for accuracy):**
  - **Test Accuracy:** 86.14%

- **Very Late Shipment Model (optimized for recall):**
  - **Test Recall:** 97.58%
  - **Average Precision Score:** 95.49% (Threshold: 0.3)

The late model provides broad classification coverage of delayed shipments, while the very late model prioritizes capturing as many high-risk cases as possible. Together, they support more proactive and targeted logistics interventions.

## Project Architecture

The repository follows a modular, production-ready structure designed for clarity, reproducibility, and scalability. Key components include data preprocessing scripts, model training modules, a containerized FastAPI application, and automated tests. Below is an overview of the file and folder organization:

```text
late-shipment-predictions-ml/
│
├── api/                     # FastAPI application and endpoint logic
│   ├── main.py
│   ├── shipment_schema.py
│
├── data/                            # Data storage directory
│   ├── raw/                         # Contains raw shipment data (included)
│   │   └── shipments_raw.csv        # Main dataset used as pipeline input
│   ├── unprocessed/                 # Generated intermediate data after cleaning
│   │   └── X_unprocessed.pkl        # Data snapshot before encoding and scaling
│   ├── preprocessed/                # Final features used for model training
│   │   ├── X_train.pkl
│   │   ├── X_test.pkl
│   │   ├── y_late_train.pkl
│   │   ├── y_late_test.pkl
│   │   ├── y_very_late_train.pkl
│   │   └── y_very_late_test.pkl
│   └── docs/                        # Supporting documentation
│       └── variable_description.csv # Column descriptions and definitions
│
├── logs/
│   └── pipeline.log         # Stores logs from model pipeline runs
│
├── mlruns/                  # MLflow experiment tracking (created during tuning, not included by default)
│
├── models/                  # Trained ML models and preprocessing artifacts
│   ├── late_model.pkl          # Random Forest model predicting late shipments (1+ days)
│   ├── very_late_model.pkl     # Random Forest model predicting very late shipments (3+ days)
│   ├── onehot_encoder.pkl      # Encoder for nominal categorical features
│   ├── ordinal_encoder.pkl     # Encoder for ordinal categorical features
│   └── scaler.pkl              # Scaler for numeric feature normalization
│
├── notebooks/               # Exploratory Data Analysis (EDA)
│   └── eda.ipynb                # Initial data exploration, feature trends, and target imbalance visualization
│
├── routers/                 # FastAPI route definitions and integration test
│   ├── landing.py              # Defines the root ("/") endpoint with a landing page message
│   ├── ping.py                 # Health check endpoint ("/ping") for uptime monitoring
│   ├── predict_late.py         # Endpoint for predicting late shipments (1+ day delay)
│   └── predict_very_late.py    # Endpoint for predicting very late shipments (3+ day delay)
│
├── src/                     # Core logic for the machine learning pipeline
│   ├── load_data.py              # Loads raw shipment data from CSV into a DataFrame
│   ├── clean_data.py             # Cleans missing values, handles duplicates, and filters irrelevant rows
│   ├── feature_engineering.py    # Generates predictive features (e.g., shipping duration, delivery gaps)
│   ├── preprocess_features.py    # Splits data, encodes categorical variables, scales features, and saves transformers
│   ├── train_late_model.py       # Trains Random Forest classifier to predict late shipments (optimized for accuracy)
│   ├── train_very_late_model.py  # Trains separate Random Forest classifier for very late shipments (optimized for recall)
│   └── logger.py                 # Centralized logger for consistent logging across all modules
│
├── tests/                   # Pytest scripts for testing API endpoints
│   ├── test_main.py             # Tests root landing page and /ping health check endpoint
│   └── test_predict_very_late.py # Tests /predict_very_late route (3+ day delay)
│
├── tuning/                  # Model tuning scripts with MLflow experiment tracking
│   ├── tune_late_model.py       # Tunes Random Forest for predicting 1+ day late shipments (optimized for accuracy)
│   └── tune_very_late_model.py  # Tunes Random Forest for 3+ day late shipments (optimized for recall)
│
├── run_pipeline.py          # Main pipeline script to execute the ML workflow
├── requirements.txt         # List of dependencies
├── Dockerfile               # Used to containerize and deploy the FastAPI app
└── README.md
```

## Installation and Running the Pipeline

This project includes a complete **machine learning pipeline** for predicting shipment delays. The pipeline prepares the data and trains two models: one for **"late"** deliveries and one for **"very late"** deliveries.

Key pipeline features:
- Fully automated script: `run_pipeline.py`
- Cleans and transforms raw shipment data
- Trains two separate classification models
- Saves preprocessing tools and model files to the `models/` directory
- Console logging for easy progress tracking

**To get started:**
1. Make sure Python 3.11+ is installed on your system.
2. (Optional but recommended) Create and activate a virtual environment.
3. Install dependencies by running:
   ```bash
   pip install -r requirements.txt
4. Execute the pipeline script:
   python run_pipeline.py

**Note:**
Running the pipeline will generate two model files:
- models/very_late_model.pkl
- models/late_model.pkl (excluded from the deployed app due to size constraints)

These models are used by the FastAPI app for inference via the /predict_very_late and (optionally) /predict_late endpoints.

## Deployment

This project includes a deployable **REST API** built with **FastAPI**, allowing users to interact with a trained machine learning model via HTTP requests.

Key deployment features:
- **Deployed to Render** using a **Docker container**
- **FastAPI** application serves a prediction endpoint for the **"very late"** shipment model
- **Interactive API documentation** available via **Swagger UI** at `/docs`
- **Schema validation** is implemented using Pydantic models to ensure input correctness
- A **/ping** route is included for uptime and health monitoring

**Note:**  
Due to memory limitations on the free Render tier, only the **"very late"** model is included in the app by default. The **"late"** model endpoint (`/predict_late`) is excluded from both the deployed app and the local version to ensure compatibility with constrained environments.

However, if you would like to make the `/predict_late` route available locally, you can do so by:
1. Running `run_pipeline.py` to generate the `models/late_model.pkl` file.
2. Un-commenting the following line in `main.py`:
   ```python
   # app.include_router(predict_late.router)

## Using the Deployed API

To try out the deployed FastAPI app on Render:

1. Open the Swagger UI in your browser:
   [Try the deployed API here](https://late-shipment-prediction-ml.onrender.com/docs)

2. Locate the `/predict_very_late/` endpoint.

3. Click **"Try it out"**.

4. Paste the following sample JSON into the request body field:

```json
{
  "order_item_quantity": 4,
  "order_item_total": 181.92,
  "product_price": 49.97,
  "year": 2015,
  "month": 4,
  "day": 21,
  "order_value": 737.65,
  "unique_items_per_order": 4,
  "order_item_discount_rate": 0.09,
  "units_per_order": 11,
  "order_profit_per_order": 89.13,
  "type": "DEBIT",
  "customer_segment": "Home Office",
  "shipping_mode": "Standard Class",
  "category_id": 46,
  "customer_country": "EE. UU.",
  "customer_state": "MA",
  "department_id": 7,
  "order_city": "San Pablo de las Salinas",
  "order_country": "México",
  "order_region": "Central America",
  "order_state": "México"
}
```

5. Click **"Execute"**.

6. Scroll down to the **Response Body** to view the model prediction:
   - `0` = Not very late (less than 3 days)
   - `1` = Very late (3 or more days)

Note: This endpoint uses a Random Forest classifier optimized for recall to flag high-risk orders.

## Testing the API Locally

You can run local integration tests on the API using pytest. These tests verify that:

- The landing page ("/") returns an expected welcome message
- The health check endpoint ("/ping") returns a status response
- The prediction endpoint for "very late" model returns valid results

Instructions:

1. Open your terminal or command prompt.

2. Navigate to the root of the project directory. Example:
   cd path/to/late_shipment_predictions_ml

3. Make sure required packages are installed. If not, run:
   pip install -r requirements.txt

4. Run the pipeline script to generate trained model files:
   python run_pipeline.py

5. Run pytest to execute all tests:
   pytest

Notes:

- All tests are located in the "tests/" folder.
- test_main.py checks the landing page and health check endpoints.
- test_predict_late.py was excluded from this version of the app due to memory limitations but can be restored if the late model is included.
- test_predict_very_late.py checks the /predict_very_late endpoint.
- Tests will fail if the required models are not generated before testing.
