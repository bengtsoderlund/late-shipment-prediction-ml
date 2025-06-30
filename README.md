# Predicting Late Shipments: An End-to-End Machine Learning Project

## Project Overview
Timely delivery is a critical factor in supply chain management, as late shipments can lead to customer dissatisfaction, revenue loss, and increased operational costs. In this project, I use machine learning to develop predictive models for a global sports and outdoor equipment retailer to proactively identify high-risk shipments before delays occur.

The project is designed following best practices in modern data science and machine learning development:

- **Modular pipeline**: Cleanly separated stages for loading, cleaning, feature engineering, preprocessing, training, and evaluation.
- **MLflow integration**: Tracks experiments and hyperparameter tuning for reproducibility.
- **REST API deployment**: Trained models are served using **FastAPI**, packaged in a **Docker** container, and deployed via **Render**.
- **Automated testing**: FastAPI routes are verified using **pytest**, including landing page, health check, and prediction endpoints.
- **Structured logging**: Unified logging system records progress and errors for easier debugging and traceability.
- **Exploratory data analysis (EDA)**: Initial insights and feature selection decisions are documented in a dedicated Jupyter notebook.

The project is built with a **clean codebase** and a **professional project structure**, ensuring clarity, reproducibility, and scalability.

## Dataset Information
The data for this project is provided by DataCo and is publicly available on Kaggle ([link](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)). It contains approximately 180,000 transactions that are shipped to 164 unique countries over the span of three years. The dataset provides a comprehensive view of supply chain operations, including:
- **Order details:** Order IDs, order destination, and order dates.
- **Product information:** Product IDs, pricing, and sales.
- **Customer data:** Customer segment and locations.
- **Shipping records:** Scheduled vs. actual delivery times, shipping mode, and delivery status.

## Business Problem
The dataset reveals that 57% of completed shipments are late by at least one day, and more than 7% are delayed by three or more days. To improve on-time delivery rates, businesses need to identify high-risk shipments early and take proactive measures to mitigate delays.

## Solution Approach
To address the challenge of shipment delays, this project builds machine learning models that classify whether an order will arrive late or very late. These predictive tools help prioritize shipments that require early intervention or adjusted logistics handling.

Two Random Forest classifiers were trained:
1. **Late Order Model (Optimized for Accuracy)**  
   Predicts whether an order will be delivered late (by at least one day).

2. **Very Late Order Model (Optimized for Recall)**  
   Predicts whether an order will be three or more days late prioritizing recall to flag high-risk shipments early in the process.

## Project Architecture

The repository follows a modular, production-ready structure designed for clarity, reproducibility, and scalability. Key components include data preprocessing scripts, model training modules, a containerized FastAPI application, and automated tests. Below is an overview of the file and folder organization:

late_shipment_predictions_ml/
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
│   └── docs/                        # Metadata and supporting documentation
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
│   ├── test_predict_late.py     # Tests /predict_late route (1+ day delay)
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

## Deployment

This project includes a deployable **REST API** built with **FastAPI**, allowing users to interact with trained machine learning models via HTTP requests.

Key deployment features:
- **Deployed to Render** using a **Docker container**
- **FastAPI** application serves prediction endpoints for both "late" and "very late" shipment models
- **Interactive API documentation** available via **Swagger UI** at `/docs`
- **Schema validation** is implemented using Pydantic models to ensure input correctness
- A **/ping** route is included for uptime and health monitoring

**Note:** Due to memory limitations on the free Render tier, only the **"very late"** model is hosted remotely. The **"late"** model is available locally and can be served from the same FastAPI app if deployed in a higher-resource environment.

To prevent deployment issues, the `/predict_late` endpoint is **conditionally registered** only if the file `models/late_model.pkl` exists. This model is not uploaded to GitHub or deployed to Render due to its large size. However, when running the project **locally**, the model is automatically generated by `run_pipeline.py`, and the endpoint will appear in the local Swagger UI.

## Using the Deployed API

To try out the deployed FastAPI app on Render:

1. Open the Swagger UI in your browser:
   [Try the deployed API here](https://late-shipment-prediction-ml.onrender.com/docs)

2. Locate the `/predict_very_late/` endpoint.

3. Click **"Try it out"**.

4. Paste the following sample JSON into the request body field:

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

5. Click **"Execute"**.

6. Scroll down to the **Response Body** to view the model prediction:
   - `0` = Not very late (less than 3 days)
   - `1` = Very late (3 or more days)

Note: This endpoint uses a Random Forest classifier optimized for recall to flag high-risk orders.

## Testing the API Locally

You can run local integration tests on the API using pytest. These tests verify that:

- The landing page ("/") returns an expected welcome message
- The health check endpoint ("/ping") returns a status response
- The prediction endpoints for both the "late" and "very late" models return valid results

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
- test_predict_late.py checks the /predict_late endpoint.
- test_predict_very_late.py checks the /predict_very_late endpoint.
- Tests will fail if the required models are not generated before testing.



