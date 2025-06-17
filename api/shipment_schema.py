from pydantic import BaseModel

class ShipmentFeatures(BaseModel):
    """
    Schema for incoming shipment data used for predictions.

    Field names and data types are based on the unprocessed dataset
    saved during the ML pipeline (see: 'data/unprocessed/X_unprocessed.pkl').

    Includes order, product, customer, and shipping-related inputs.
    """
    
    order_item_quantity: int
    order_item_total: float
    product_price: float
    year: int
    month: int
    day: int
    order_value: float
    unique_items_per_order: int
    order_item_discount_rate: float
    units_per_order: int
    order_profit_per_order: float
    type: str
    customer_segment: str
    shipping_mode: str
    category_id: int
    customer_country: str
    customer_state: str
    department_id: int
    order_city: str
    order_country: str
    order_region: str
    order_state: str