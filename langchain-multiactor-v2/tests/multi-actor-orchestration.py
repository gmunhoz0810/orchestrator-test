from typing import Annotated, List, Dict, Any, TypedDict
from langgraph.graph import StateGraph, Graph, END, START
from langchain_core.messages import BaseMessage, HumanMessage
import operator
from base import CodeInterpreterAgent
import functools
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Define state types for our graph
class GraphState(TypedDict):
    """Type for the graph state."""
    messages: Annotated[List[BaseMessage], operator.add]
    next: str
    current_actor: str
    shared_thread: str

class CustomerActor(CodeInterpreterAgent):
    """Specialized actor for customer data analysis."""
    def __init__(self):
        super().__init__(
            name="Customer Data Specialist",
            instructions="""You are a customer data specialist who excels at analyzing customer information.
            You have access to the customers-10000 dataset which contains detailed customer information.
            When analyzing:
            1. Focus on customer demographics, behavior, and patterns
            2. Create clear visualizations when relevant
            3. Always explain your findings in detail
            4. Pay attention to relationships that might connect to organization data
            
            Important: You only have access to customer data. If organization information is needed,
            mention this in your response.""",
            files=["data/customers-10000.csv"],
            verbose=True
        )

class OrganizationActor(CodeInterpreterAgent):
    """Specialized actor for organization data analysis."""
    def __init__(self):
        super().__init__(
            name="Organization Data Specialist",
            instructions="""You are an organization data specialist who excels at analyzing company information.
            You have access to the organizations-10000 dataset which contains detailed company information.
            When analyzing:
            1. Focus on organization types, sectors, and characteristics
            2. Create clear visualizations when relevant
            3. Always explain your findings in detail
            4. Pay attention to relationships that might connect to customer data
            
            Important: You only have access to organization data. If customer information is needed,
            mention this in your response.""",
            files=["data/organizations-10000.csv"],
            verbose=True
        )

class Orchestrator:
    """Main orchestrator class for managing multiple actors."""
    
    def __init__(self):
        self.actors = {
            "customer_specialist": CustomerActor(),
            "organization_specialist": OrganizationActor()
        }
        self.graph = self._create_graph()
        
    def _create_graph(self) -> Graph:
        """Create the graph for actor orchestration."""
        workflow = StateGraph(GraphState)
        
        # Add nodes for each actor
        for actor_name, actor in self.actors.items():
            workflow.add_node(actor_name, self._create_actor_node(actor, actor_name))
        
        # Add supervisor node
        workflow.add_node("supervisor", self._supervisor_node())
        
        # Add edges
        for actor_name in self.actors:
            workflow.add_edge(actor_name, "supervisor")
        
        # Add conditional edges from supervisor to actors
        workflow.add_conditional_edges(
            "supervisor",
            self._route_next,
            {
                "customer_specialist": "customer_specialist",
                "organization_specialist": "organization_specialist",
                "FINISH": END
            }
        )
        
        # Add starting edge
        workflow.add_edge(START, "supervisor")
        
        return workflow.compile()
    
    def _create_actor_node(self, actor: CodeInterpreterAgent, name: str):
        """Create a node for an actor in the graph."""
        def node_func(state: GraphState) -> Dict:
            # Use the same thread if it exists
            thread_id = state.get("shared_thread")
            
            # Get the last message
            last_message = state["messages"][-1].content
            
            if self.verbose:
                print(f"\n{name} processing: {last_message[:100]}...")
            
            # Run the analysis
            response = actor.run_analysis(last_message, thread_id=thread_id)
            
            # Update state
            return {
                "messages": [HumanMessage(content=response["content"], name=name)],
                "shared_thread": response.get("thread_id", thread_id)
            }
        
        return node_func
    
    def _supervisor_node(self):
        """Create the supervisor node that routes between actors."""
        def get_next_actor(state: GraphState) -> Dict[str, str]:
            last_message = state["messages"][-1].content.lower()
            last_actor = state["messages"][-1].name if state["messages"] else None
            
            # Check for completion indicators
            if any(phrase in last_message for phrase in [
                "analysis complete",
                "task finished",
                "final recommendation",
                "conclusion reached"
            ]):
                return {"next": "FINISH"}
            
            # Route based on content and context
            if "organization" in last_message or "company" in last_message:
                if last_actor != "organization_specialist":
                    return {"next": "organization_specialist"}
                
            if "customer" in last_message or "individual" in last_message:
                if last_actor != "customer_specialist":
                    return {"next": "customer_specialist"}
            
            # If no clear routing, alternate between specialists
            if last_actor == "customer_specialist":
                return {"next": "organization_specialist"}
            else:
                return {"next": "customer_specialist"}
        
        return get_next_actor
    
    def _route_next(self, state: GraphState) -> str:
        """Route to the next actor based on supervisor decision."""
        return state["next"]
    
    def run(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        """Run the orchestrator with a query."""
        self.verbose = verbose
        
        if verbose:
            print("\nInitiating analysis with query:", query)
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "next": "",
            "current_actor": "",
            "shared_thread": ""
        }
        
        # Run the graph
        try:
            result = self.graph.invoke(initial_state)
            
            if verbose:
                print("\nAnalysis completed successfully!")
                
            return result
            
        finally:
            # Clean up resources
            if verbose:
                print("\nCleaning up resources...")
            for actor in self.actors.values():
                actor.cleanup()

# Example usage
if __name__ == "__main__":
    orchestrator = Orchestrator()
    
    # Simpler query that requires both datasets but no visualizations
    query = """Help me understand the relationship between organizations and customers.
    Specifically:
    1. From the customer data, tell me the top 3 organizations by number of customers.
    2. From the organization data, tell me the details about these top 3 organizations.
    
    Please focus on a plain text analysis without generating any visualizations."""
    
    result = orchestrator.run(query, verbose=True)
    
    print("\nFinal Analysis Result:")
    print(result)