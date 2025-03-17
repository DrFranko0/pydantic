from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
from models.support import OrderInfo
from config import MODEL_NAME

@dataclass
class OrderStatusDependencies:
    order_id: str
    customer_id: str

order_status_tool = Agent(
    f"gemini:{MODEL_NAME}",
    deps_type=OrderStatusDependencies,
    result_type=OrderInfo,
    system_prompt="""
    You are an order status lookup tool that simulates access to an order database.
    Given an order ID and customer ID, return information about the order.
    
    For this simulation:
    - If order ID starts with "A", return a status of "Delivered"
    - If order ID starts with "B", return a status of "Shipped"
    - If order ID starts with "C", return a status of "Processing"
    - If order ID starts with "D", return a status of "Cancelled"
    - Otherwise, return a status of "Not Found"
    
    Include reasonable values for all other fields.
    """
)

@order_status_tool.tool
async def check_order_status(ctx: RunContext[OrderStatusDependencies]) -> OrderInfo:
    
    prompt = f"Look up order status for Order ID: {ctx.deps.order_id} and Customer ID: {ctx.deps.customer_id}"
    
    result = await order_status_tool.run(prompt)
    return result.data
