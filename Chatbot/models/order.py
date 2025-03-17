from pydantic import BaseModel, Field
from typing import List, Optional

class ProductInfo(BaseModel):
    product_id: str = Field(description="Unique identifier for the product")
    name: str = Field(description="Product name")
    category: str = Field(description="Product category")
    price: float = Field(description="Product price")
    in_stock: bool = Field(description="Whether the product is in stock")
    
class OrderInfo(BaseModel):
    order_id: str = Field(description="Unique identifier for the order")
    customer_id: str = Field(description="Customer ID who placed the order")
    status: str = Field(description="Current status of the order")
    items: List[str] = Field(description="List of items in the order")
    total: float = Field(description="Total order amount")
    date: str = Field(description="Order date")
    estimated_delivery: Optional[str] = Field(None, description="Estimated delivery date if applicable")

class CustomerRecord(BaseModel):
    customer_id: str = Field(description="Unique identifier for the customer")
    name: str = Field(description="Customer name")
    email: str = Field(description="Customer email")
    phone: Optional[str] = Field(None, description="Customer phone number")
    membership_level: str = Field(description="Customer membership level")
    account_age_days: int = Field(description="Number of days the customer has had an account")