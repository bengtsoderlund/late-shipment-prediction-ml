"""
tests/test_predict_very_late.py

Integration test for the /predict_very_late/ endpoint.
"""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_very_late():
    input_data = {
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
    
    response = client.post("/predict_very_late", json=input_data)
    
    assert response.status_code == 200
    result = response.json()
    assert "very_late_prediction" in result
    assert isinstance(result["very_late_prediction"], int)


