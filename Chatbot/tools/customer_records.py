from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
from models.order import CustomerRecord
from config import MODEL_NAME

@dataclass
class CustomerRecordDependencies:
    customer_id: str

customer_records_tool = Agent(
    f"gemini:{MODEL_NAME}",
    deps_type=CustomerRecordDependencies,
    result_type=CustomerRecord,
    system_prompt="""
    You are a customer record lookup tool that simulates access to a customer database.
    Given a customer ID, return information about the customer.
    
    For this simulation:
    - If customer ID is "C12345", return a customer named "John Doe"
    - If customer ID is "C67890", return a customer named "Jane Smith"
    - Otherwise, generate a reasonable customer record
    
    Include reasonable values for all fields.
    """
)

@customer_records_tool.tool
async def lookup_customer(ctx: RunContext[CustomerRecordDependencies]) -> CustomerRecord:
    
    prompt = f"Look up customer information for Customer ID: {ctx.deps.customer_id}"
    
    result = await customer_records_tool.run(prompt)
    return result.data
