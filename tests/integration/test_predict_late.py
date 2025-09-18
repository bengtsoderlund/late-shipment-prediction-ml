"""
tests/integration/test_predict_late.py

Integration test for the /predict_late endpoint using real S3-loaded artifacts.
"""

import pytest
pytestmark = pytest.mark.integration

def test_late(client):
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
    
    resp = client.post("/predict_late/", json=input_data)
    
    assert resp.status_code == 200
    body = resp.json()
    assert "late_prediction" in body
    assert body["late_prediction"] in (0, 1)