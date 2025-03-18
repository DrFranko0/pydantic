import asyncio
import typer
import os
from dotenv import load_dotenv

from graph.workflow import ConversationGraph
from utils.message_history import MessageHistoryManager
from utils.human_handoff import HumanHandoffManager

load_dotenv()

app = typer.Typer()
history_manager = MessageHistoryManager()
handoff_manager = HumanHandoffManager()

@app.command("chat")
def chat(customer_id: str = typer.Option("C12345", help="Customer ID")):
    typer.echo(f"Starting chat for customer {customer_id}")
    typer.echo("Type 'exit' to end the conversation")
    
    conversation = ConversationGraph(customer_id=customer_id)
    
    typer.echo("\nAI Support: Hello! How can I help you today?")
    
    while True:
        user_input = typer.prompt("You")
        
        if user_input.lower() == "exit":
            break
        
        result = asyncio.run(conversation.process_message(user_input))
        
        if result.get("needs_human", False):
            typer.echo("\nConversation has been escalated to a human agent.")
            handoff_id = handoff_manager.queue_handoff(customer_id, result)
            typer.echo(f"Handoff ID: {handoff_id}")
            break
  
        if result.get("resolved", False):
            typer.echo("\nThe conversation has been resolved.")
            break
        
        history = conversation.get_conversation_history()
        if history and history[-1].role == "assistant":
            typer.echo(f"\nAI Support: {history[-1].content}")
    
    filepath = history_manager.save_conversation(customer_id, conversation.get_conversation_history())
    typer.echo(f"\nConversation saved to {filepath}")

@app.command("handoffs")
def list_handoffs():
    handoffs = handoff_manager.get_pending_handoffs()
    
    if not handoffs:
        typer.echo("No pending handoffs")
        return
    
    typer.echo(f"Found {len(handoffs)} pending handoffs:")
    for handoff in handoffs:
        typer.echo(f"ID: {handoff['handoff_id']}, Customer: {handoff['customer_id']}, Reason: {handoff['escalation_reason']}")

@app.command("view-handoff")
def view_handoff(handoff_id: str):
    handoff = handoff_manager.get_handoff_details(handoff_id)
    
    if not handoff:
        typer.echo(f"Handoff {handoff_id} not found")
        return
    
    typer.echo(f"Handoff details for {handoff_id}:")
    typer.echo(f"Customer ID: {handoff['customer_id']}")
    typer.echo(f"Escalation Reason: {handoff['escalation_reason']}")
    typer.echo(f"Sentiment: {handoff['sentiment']}")
    typer.echo(f"Status: {handoff['status']}")
    
    typer.echo("\nConversation:")
    for message in handoff['conversation']:
        prefix = "Customer" if message.role == "user" else "AI Support"
        typer.echo(f"{prefix}: {message.content}")

if __name__ == "__main__":
    app()
