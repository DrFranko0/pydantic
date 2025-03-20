import streamlit as st
import asyncio
import logfire
from main import run_research

logfire.configure()
logger = logfire.getLogger("streamlit_app")

st.set_page_config(
    page_title="Autonomous Research Assistant",
    layout="wide"
)

st.title("Autonomous Research Assistant")
st.markdown("""
This application helps you research any topic by:
1. Generating focused research questions
2. Searching for relevant information
3. Analyzing and extracting key findings
4. Synthesizing a comprehensive research report
""")

st.header("Research Topic")
topic = st.text_input("Enter the topic you want to research", "")
start_button = st.button("Start Research")

progress_placeholder = st.empty()
result_placeholder = st.empty()

async def run_with_progress(topic):
    progress_bar = progress_placeholder.progress(0)
    
    status_message = result_placeholder.info("Starting research...")
    
    try:
        for i, step in enumerate(["Generating questions", "Searching for information", 
                               "Analyzing content", "Synthesizing findings"], 1):
            progress_bar.progress(i * 20)
            status_message.info(f"Step {i}/5: {step}...")
            await asyncio.sleep(0.5) 
        
        report = await run_research(topic)
        
        progress_bar.progress(100)
        
        if report:
            status_message.success("Research completed!")
            result_placeholder.markdown(report)
            
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

if start_button and topic:
    asyncio.run(run_with_progress(topic))

st.sidebar.header("About")
st.sidebar.markdown("""
This research assistant uses:
- PydanticAI with Gemini LLM models
- Graph-based workflow orchestration
- Web search capabilities
- Automatic citation generation

Enter any topic of interest and get a comprehensive research report.
""")
