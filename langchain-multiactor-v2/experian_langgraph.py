from typing import Annotated, Dict, List, TypedDict, Union
import functools
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from openai import AzureOpenAI
from datetime import datetime
import json
import os
from dotenv import load_dotenv
import sys
import time

# Load environment variables from .env file
load_dotenv()

# Get Azure OpenAI credentials from environment
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

if not AZURE_OPENAI_KEY or not AZURE_OPENAI_ENDPOINT:
    print("Error: Azure OpenAI credentials not found in .env file")
    sys.exit(1)

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-02-15-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

print(f"[{datetime.now()}] 🔑 Successfully loaded Azure OpenAI credentials")

def create_assistant_with_file(name: str, instructions: str, file_path: str) -> Dict:
    """Create an Azure OpenAI assistant with code interpreter and file access."""
    print(f"\n[{datetime.now()}] Creating assistant: {name}")
    
    # Upload file
    file = client.files.create(
        file=open(file_path, "rb"),
        purpose='assistants'
    )
    print(f"[{datetime.now()}] Uploaded file: {file_path} (ID: {file.id})")
    
    # Create assistant with code interpreter and file access
    assistant = client.beta.assistants.create(
        name=name,
        instructions=instructions,
        model="gpt-4",  # Use your Azure OpenAI deployment name
        tools=[{"type": "code_interpreter"}],
        tool_resources={
            "code_interpreter": {
                "file_ids": [file.id]
            }
        }
    )
    print(f"[{datetime.now()}] Created assistant: {name} (ID: {assistant.id})")
    return {"assistant_id": assistant.id, "file_id": file.id}

def create_thread_and_run(assistant_id: str, file_id: str, user_message: str) -> Dict:
    """Create a thread and run for an assistant, returning the results."""
    print(f"\n[{datetime.now()}] Creating thread for assistant: {assistant_id}")
    
    # Create thread
    thread = client.beta.threads.create()
    print(f"[{datetime.now()}] Created thread: {thread.id}")
    
    # Add message to thread
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_message
    )
    print(f"[{datetime.now()}] Added message to thread")
    
    # Create run
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id
    )
    print(f"[{datetime.now()}] Created run: {run.id}")
    
    # Wait for completion
    print(f"[{datetime.now()}] Waiting for run completion...")
    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run.status == 'completed':
            print(f"[{datetime.now()}] Run completed!")
            break
        elif run.status == 'failed':
            raise Exception(f"Run failed: {run.last_error}")
        elif run.status == 'requires_action':
            print(f"[{datetime.now()}] Run requires action: {run.required_action}")
        print(f"[{datetime.now()}] Current status: {run.status}")
        time.sleep(1)
    
    # Get messages
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    result = messages.data[0].content[0].text.value
    
    print(f"\n[{datetime.now()}] Assistant {assistant_id} response:")
    print("=" * 50)
    print(result)
    print("=" * 50)
    
    return {"thread_id": thread.id, "run_id": run.id, "result": result}

# Create our assistants
customer_assistant = create_assistant_with_file(
    "Customer Data Analyst",
    "You are a customer data analyst. Use code interpreter to analyze customer data. Always explain your analysis process.",
    "data/customers-10000.csv"
)

corporation_assistant = create_assistant_with_file(
    "Corporation Data Analyst",
    "You are a corporation data analyst. Use code interpreter to analyze organization data. Always explain your analysis process.",
    "data/organizations-10000.csv"
)

class AnalysisState(TypedDict):
    """State for our analysis workflow."""
    messages: Annotated[List[BaseMessage], operator.add]
    next: str
    results: Dict[str, str]
    orchestrator_summary: str


def analyze_customer_data(state: AnalysisState) -> AnalysisState:
    """Node for analyzing customer data."""
    print(f"\n[{datetime.now()}] 👤 CUSTOMER DATA ANALYSIS NODE")
    last_message = state["messages"][-1].content
    
    result = create_thread_and_run(
        customer_assistant["assistant_id"],
        customer_assistant["file_id"],
        last_message
    )
    
    return {
        **state,
        "messages": [HumanMessage(content=result["result"], name="CustomerAnalyst")],
        "results": {**state.get("results", {}), "customer": result["result"]}
    }

def analyze_corporation_data(state: AnalysisState) -> AnalysisState:
    """Node for analyzing corporation data."""
    print(f"\n[{datetime.now()}] 🏢 CORPORATION DATA ANALYSIS NODE")
    last_message = state["messages"][-1].content
    
    result = create_thread_and_run(
        corporation_assistant["assistant_id"],
        corporation_assistant["file_id"],
        last_message
    )
    
    return {
        **state,
        "messages": [HumanMessage(content=result["result"], name="CorporationAnalyst")],
        "results": {**state.get("results", {}), "corporation": result["result"]}
    }

def create_orchestrator(llm: ChatOpenAI):
    """Create the orchestrator that decides which analysts to use."""
    options = ["CUSTOMER_DATA", "CORPORATION_DATA", "SUMMARIZE", "FINISH"]
    
    function_def = {
        "name": "route",
        "description": "Route to the next action",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                },
                "reasoning": {
                    "title": "Reasoning",
                    "type": "string",
                    "description": "Explanation for this routing decision"
                }
            },
            "required": ["next", "reasoning"],
        },
    }
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a data analysis orchestrator that routes questions to appropriate specialists:
        - CUSTOMER_DATA: For questions about customer information
        - CORPORATION_DATA: For questions about organization information
        - SUMMARIZE: When you have enough information to create a final summary
        - FINISH: When the analysis is complete
        
        You can call multiple analysts before summarizing. Track what information you've received and what's still needed.
        """),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Based on the conversation above, what should be the next action? Select from: {options}")
    ]).partial(options=str(options))
    
    return (prompt
            | llm.bind_tools(tools=[function_def])  # Updated to use bind_tools
            | JsonOutputFunctionsParser())

def summarize_results(state: AnalysisState) -> AnalysisState:
    """Create a final summary of all analysis results."""
    print(f"\n[{datetime.now()}] 📊 SUMMARIZING RESULTS")
    
    summary_prompt = f"""Based on the following analysis results, provide a comprehensive summary:
    
    Customer Analysis:
    {state.get('results', {}).get('customer', 'No customer analysis performed')}
    
    Corporation Analysis:
    {state.get('results', {}).get('corporation', 'No corporation analysis performed')}
    
    Provide a clear, concise summary that combines these insights.
    """
    
    result = create_thread_and_run(
        customer_assistant["assistant_id"],
        customer_assistant["file_id"],
        summary_prompt
    )
    
    print(f"\n[{datetime.now()}] 🎯 FINAL SUMMARY:")
    print("=" * 50)
    print(result["result"])
    print("=" * 50)
    
    return {
        **state,
        "messages": [HumanMessage(content=result["result"], name="Summarizer")],
        "orchestrator_summary": result["result"],
        "next": "FINISH"
    }

def create_analysis_workflow(llm: ChatOpenAI):
    """Create the full analysis workflow."""
    workflow = StateGraph(AnalysisState)
    
    orchestrator = create_orchestrator(llm)
    
    # Add nodes
    workflow.add_node("customer_analyst", analyze_customer_data)
    workflow.add_node("corporation_analyst", analyze_corporation_data)
    workflow.add_node("orchestrator", orchestrator)
    workflow.add_node("summarizer", summarize_results)
    
    # Add edges
    workflow.add_edge("customer_analyst", "orchestrator")
    workflow.add_edge("corporation_analyst", "orchestrator")
    workflow.add_edge("summarizer", "orchestrator")
    
    # Add conditional edges from orchestrator
    workflow.add_conditional_edges(
        "orchestrator",
        lambda x: x["next"],
        {
            "CUSTOMER_DATA": "customer_analyst",
            "CORPORATION_DATA": "corporation_analyst",
            "SUMMARIZE": "summarizer",
            "FINISH": END
        }
    )
    
    workflow.add_edge(START, "orchestrator")
    
    return workflow.compile()

def analyze_data(question: str, llm: ChatOpenAI = ChatOpenAI(model="gpt-4")):
    """Run the analysis workflow for a given question."""
    chain = create_analysis_workflow(llm)
    
    print(f"\n[{datetime.now()}] 🚀 Starting analysis for question: {question}")
    print("=" * 50)
    
    result = chain.invoke({
        "messages": [HumanMessage(content=question)],
        "results": {},
        "next": "",
        "orchestrator_summary": ""
    })
    
    return result

if __name__ == "__main__":
    question = "How many customers do we have and what's their relationship with our corporations?"
    result = analyze_data(question)
    
    print("\n🎉 Analysis Complete!")
    print("\nFinal Answer for User:")
    print("=" * 50)
    print(result["orchestrator_summary"])
    print("=" * 50)