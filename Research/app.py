import streamlit as st
import asyncio
import logfire
from main import run_research

# Configure logger
logfire.configure()
logger = logfire.getLogger("streamlit_app")

# Configure page
st.set_page_config(
    page_title="Autonomous Research Assistant",
    layout="wide"
)

# Main title
st.title("Autonomous Research Assistant")
st.markdown("""
This application helps you research any topic by:
1. Generating focused research questions
2. Searching for relevant information
3. Analyzing and extracting key findings
4. Synthesizing a comprehensive research report
""")

# Input section
st.header("Research Topic")
topic = st.text_input("Enter the topic you want to research", "")
start_button = st.button("Start Research")

# Create placeholders
progress_placeholder = st.empty()
result_placeholder = st.empty()

# Define async function to update progress
async def run_with_progress(topic):
    # Create progress bar
    progress_bar = progress_placeholder.progress(0)
    
    # Display status message
    status_message = result_placeholder.info("Starting research...")
    
    try:
        # Update progress during key steps
        for i, step in enumerate(["Generating questions", "Searching for information", 
                               "Analyzing content", "Synthesizing findings"], 1):
            progress_bar.progress(i * 20)
            status_message.info(f"Step {i}/5: {step}...")
            await asyncio.sleep(0.5)  # Small delay for UI updates
        
        # Call research function
        report = await run_research(topic)
        
        # Update progress to 100%
        progress_bar.progress(100)
        
        # Display result
        if report:
            status_message.success("Research completed!")
            result_placeholder.markdown(report)
            
            # Add download button
            st.download_button(
                label="Download Report (Markdown)",
                data=report,
                file_name=f"{topic.replace(' ', '_')}_research_report.md",
                mime="text/markdown"
            )
        else:
            status_message.error("Research failed. Please try again.")
    
    except Exception as e:
        logger.exception("Error during research")
        progress_bar.progress(100)
        status_message.error(f"Error: {str(e)}")

# Run when button is clicked
if start_button and topic:
    asyncio.run(run_with_progress(topic))

# Sidebar with additional information
st.sidebar.header("About")
st.sidebar.markdown("""
This research assistant uses:
- PydanticAI with Gemini LLM models
- Graph-based workflow orchestration
- Web search capabilities
- Automatic citation generation

Enter any topic of interest and get a comprehensive research report.
""")
