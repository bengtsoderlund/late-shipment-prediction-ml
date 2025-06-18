"""
landing.py

Defines the root endpoint (`/`) that serves an HTML landing page 
explaining the purpose of the API and how to use it.
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def landing_page():
    return """
    <html>
        <head>
            <title>Late Shipment Prediction API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: auto; padding: 2em; line-height: 1.6; }
                h1 { color: #2c3e50; }
                h2 { color: #34495e; margin-top: 2em; }
                code { background-color: #f4f4f4; padding: 0.2em 0.4em; border-radius: 4px; }
                pre { background-color: #f8f8f8; padding: 1em; border-radius: 6px; overflow-x: auto; }
                ul { padding-left: 1.2em; }
                li { margin-bottom: 0.5em; }
                a { color: #1f6feb; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <h1>📦 Late Shipment Prediction API</h1>

            <p>
                This API uses machine learning to predict late shipments for a global sports and outdoor equipment retailer. 
                The models are trained on publicly available shipping data provided by DataCo.
            </p>

            <h2>🧠 Available Models</h2>
            <ul>
                <li>
                    <strong>Late Order Model</strong> – Predicts whether an order will be <em>late by at least 1 day</em>.<br>
                    <em>Optimized for Accuracy</em> (92.1%)<br>
                    ❌ <strong>Currently unavailable</strong> due to memory limits on Render’s free tier
                </li>
                <li>
                    <strong>Very Late Order Model</strong> – Predicts whether an order will be <em>very late (≥ 3 days)</em>.<br>
                    <em>Optimized for Recall</em> (97.3%)<br>
                    ✅ <strong>Available via this API</strong>
                </li>
            </ul>

            <h2>📋 How to Use the API</h2>
            <p>
                Use the interactive documentation to test the model with your own input data, or send a POST request to the API.
            </p>
            <p>
                ➡️ <a href="/docs"><strong>Go to Swagger UI (Interactive API Docs)</strong></a>
            </p>

            <p><strong>📤 Example JSON input:</strong> (copy and paste into Swagger UI)</p>
            <pre>
{
    "order_item_quantity": 3,
    "order_item_total": 136.44,
    "product_price": 49.97,
    "year": 2016,
    "month": 2,
    "day": 17,
    "order_value": 805.22,
    "unique_items_per_order": 5,
    "order_item_discount_rate": 0.09,
    "units_per_order": 13,
    "order_profit_per_order": 65.48,
    "type": "DEBIT",
    "customer_segment": "Corporate",
    "shipping_mode": "First Class",
    "category_id": 46,
    "customer_country": "EE. UU.",
    "customer_state": "CA",
    "department_id": 7,
    "order_city": "Adelaide",
    "order_country": "Australia",
    "order_region": "Oceania",
    "order_state": "Australia del Sur"
}
            </pre>

            <h2>📌 Notes</h2>
            <ul>
                <li>This is a POST-based API. The <code>/predict_late/</code> endpoint does not accept GET requests.</li>
                <li>You can test the endpoint directly via Swagger UI, Postman, or curl.</li>
                <li>The full "late" model may be deployed in the future by upgrading to a paid Render plan with higher memory capacity.</li>
                <li>A lightweight <code>/ping</code> endpoint is available for uptime and health checks.</li>
                <li>All code is modularized and containerized with Docker for reproducibility.</li>
            </ul>
        </body>
    </html>
    """
