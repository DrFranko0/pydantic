from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
from models.order import ProductInfo
from typing import Optional
from config import MODEL_NAME

@dataclass
class ProductLookupDependencies:
    product_id: Optional[str] = None
    product_name: Optional[str] = None

product_lookup_tool = Agent(
    f"gemini:{MODEL_NAME}",
    deps_type=ProductLookupDependencies,
    result_type=ProductInfo,
    system_prompt="""
    You are a product lookup tool that simulates access to a product database.
    Given a product ID or name, return information about the product.
    
    For this simulation:
    - If given product ID "P12345", return information about "Wireless Earbuds"
    - If given product ID "P67890", return information about "Smart Watch"
    - If given product name containing "earbuds" or "headphones", return info on "Wireless Earbuds"
    - If given product name containing "watch", return info on "Smart Watch"
    - Otherwise, return a product with reasonable default values
    """
)

@product_lookup_tool.tool
async def lookup_product(ctx: RunContext[ProductLookupDependencies]) -> ProductInfo:
    
    prompt = "Look up product information for "
    
    if ctx.deps.product_id:
        prompt += f"product ID: {ctx.deps.product_id}"
    elif ctx.deps.product_name:
        prompt += f"product name: {ctx.deps.product_name}"
    else:
        prompt += "an unknown product (return default values)"
    
    result = await product_lookup_tool.run(prompt)
    return result.data
