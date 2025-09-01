"""
landing.py

Root ("/") HTML landing page for the Shipment Delay Prediction API.
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def landing_page():
    return """
    <html>
      <head>
        <title>Shipment Delay Prediction API</title>
        <meta charset="utf-8" />
        <style>
          body { font-family: Arial, sans-serif; max-width: 900px; margin: auto; padding: 2rem; line-height: 1.6; }
          h1 { color: #1f2937; }
          h2 { color: #374151; margin-top: 1.5rem; }
          code { background: #f3f4f6; padding: 0.2rem 0.4rem; border-radius: 4px; }
          pre { background: #f8fafc; padding: 1rem; border-radius: 6px; overflow-x: auto; }
          ul { padding-left: 1.25rem; }
          li { margin-bottom: 0.4rem; }
          a { color: #1f6feb; text-decoration: none; }
          a:hover { text-decoration: underline; }
          .pill { display:inline-block; padding:0.15rem 0.5rem; border-radius:9999px; background:#eef2ff; color:#3730a3; font-size:0.85rem; margin-left:0.25rem; }
        </style>
      </head>
      <body>
        <h1>📦 Shipment Delay Prediction API</h1>

        <p>
          Production-style FastAPI service running on AWS ECS/Fargate.
          Models and preprocessing artifacts are stored in Amazon S3 and loaded at runtime.
        </p>

        <h2>🧠 Endpoints</h2>
        <ul>
          <li><code>POST /predict_late/</code> <span class="pill">late ≥ 1 day</span></li>
          <li><code>POST /predict_very_late/</code> <span class="pill">very late ≥ 3 days</span></li>
          <li><code>GET /ping</code> <span class="pill">health</span></li>
          <li><a href="/docs"><strong>Swagger UI</strong></a> <span class="pill">interactive</span></li>
        </ul>

        <h2>📋 Example JSON</h2>
        <p>Use this structure in Swagger UI for either prediction endpoint:</p>
        <pre>{
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
}</pre>

        <h2>🛠 Notes</h2>
        <ul>
          <li>POST-only for predictions; use Swagger UI to explore schema and try requests.</li>
          <li>Container logs stream to CloudWatch for observability.</li>
          <li>Least-privilege IAM Task Role grants read-only access to S3 artifacts.</li>
        </ul>

        <p>➡️ <a href="/docs"><strong>Open Swagger UI</strong></a></p>
      </body>
    </html>
    """
