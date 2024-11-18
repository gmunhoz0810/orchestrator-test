from typing import Dict, List, Optional, Tuple
from base import CodeInterpreterAgent
from actor_metadata import ActorMetadata
import time
from azure_config import get_azure_client

class OrchestratorContext:
    """Maintains context between runs"""
    def __init__(self):
        # Track threads per actor
        self.threads: Dict[str, str] = {}
        # Track latest results per actor
        self.actor_results: Dict[str, List[Dict]] = {}
        # Track conversation history
        self.history: List[Dict] = []
    
    def add_result(self, actor: str, result: Dict):
        """Add a result to actor's history"""
        if actor not in self.actor_results:
            self.actor_results[actor] = []
        self.actor_results[actor].append(result)
    
    def get_actor_history(self, actor: str) -> str:
        """Get formatted history for an actor"""
        if actor not in self.actor_results:
            return "No previous context."
        
        history = []
        for result in self.actor_results[actor]:
            history.append(f"Previous analysis: {result.get('content', 'No content')}")
        return "\n\n".join(history)

class EnhancedOrchestrator:
    """Orchestrator using Azure OpenAI for planning and execution"""
    
    def __init__(self, verbose: bool = True):
        self.client = get_azure_client()
        self.actors: Dict[str, tuple[CodeInterpreterAgent, ActorMetadata]] = {}
        self.verbose = verbose
        self.context = OrchestratorContext()
        
        # Create planning assistant
        self.planner = CodeInterpreterAgent(
            name="Planning Assistant",
            instructions="""You are an AI planner that creates efficient execution plans.
            When given a query and available actors, you will:
            1. Analyze what data and analysis is needed
            2. Create minimal, efficient execution plans
            3. Choose the most appropriate actors for each task
            4. Ensure data flows correctly between actors
            5. Return plans in the format:
               1. actor_name: specific_task
               2. actor_name: specific_task
            
            Rules:
            - Only use actors that are actually needed
            - Keep plans minimal but complete
            - Ensure each actor has necessary context
            - Consider previous results when planning""",
            model="gpt-4"  # Use your Azure deployment name
        )
        
        if verbose:
            print("Initialized orchestrator with Azure OpenAI")
    
    def log(self, message: str, level: str = "INFO"):
        """Unified logging with levels"""
        if self.verbose:
            print(f"[Orchestrator:{level}] {message}")
    
    def register_actor(self, name: str, agent: CodeInterpreterAgent, metadata: ActorMetadata):
        """Register an actor with its metadata"""
        self.actors[name] = (agent, metadata)
        self.log(f"Registered {name} with access to {metadata.file_path}")
        if self.verbose:
            self.log(f"Actor capabilities: {metadata.data_description}", "DEBUG")
    
    def _create_actor_prompt(self, actor_name: str, task: str) -> str:
        """Create a context-aware prompt for an actor"""
        history = self.context.get_actor_history(actor_name)
        other_results = {}
        
        # Get relevant results from other actors
        for other_actor, results in self.context.actor_results.items():
            if other_actor != actor_name and results:
                other_results[other_actor] = results[-1].get('content', '')
        
        prompt = f"""Task: {task}

Your Previous Context:
{history}

Other Actors' Recent Results:
{chr(10).join(f'{actor}: {result}' for actor, result in other_results.items())}

Requirements:
1. Use the file ID provided in your initialization
2. Reference previous analysis if relevant
3. Be specific and quantitative
4. Return clear, structured results
5. Focus only on your specific capabilities"""

        return prompt
    
    def _plan_execution(self, query: str) -> List[Tuple[str, str]]:
        """Create execution plan using planning assistant"""
        # Create description of available actors and their context
        actor_descriptions = []
        for name, (_, meta) in self.actors.items():
            desc = f"Actor '{name}':\n"
            desc += f"- Data: {meta.data_description}\n"
            desc += f"- Previous context: {'Yes' if name in self.context.actor_results else 'No'}\n"
            if name in self.context.actor_results:
                last_result = self.context.actor_results[name][-1].get('content', '')
                desc += f"- Latest result: {last_result[:200]}...\n"
            actor_descriptions.append(desc)
            
        planning_query = f"""Create an execution plan for this query:
        "{query}"
        
        Available actors:
        {chr(10).join(actor_descriptions)}
        
        Previous execution history:
        {chr(10).join(f'- {h}' for h in self.context.history)}
        
        Return only numbered steps in format:
        1. actor_name: task_description"""
        
        result = self.planner.run_analysis(planning_query)
        
        # Parse the steps from the response
        steps = []
        for line in result["content"].strip().split("\n"):
            if ':' in line and line.strip():
                actor, task = line.split(":", 1)
                actor = actor.strip().split(".", 1)[-1].strip()
                if actor in self.actors:
                    steps.append((actor, task.strip()))
        
        self.log("Execution plan:")
        for i, (actor, task) in enumerate(steps, 1):
            self.log(f"{i}. {actor} -> {task}")
            
        return steps
    
    def _execute_actor(self, actor_name: str, task: str) -> Dict:
        """Execute a single actor task with proper context"""
        agent, _ = self.actors[actor_name]
        self.log(f"Executing {actor_name} with task: {task}")
        
        # Create context-aware prompt
        prompt = self._create_actor_prompt(actor_name, task)
        
        # Use existing thread if available
        thread_id = self.context.threads.get(actor_name)
        result = agent.run_analysis(prompt, thread_id=thread_id)
        
        # Store thread ID if new
        if not thread_id and result.get("thread_id"):
            self.context.threads[actor_name] = result["thread_id"]
            self.log(f"Created thread for {actor_name}: {result['thread_id']}")
            
        # Store result in context
        self.context.add_result(actor_name, result)
        # Add to history
        self.context.history.append(f"{actor_name}: {task}")
        
        return result
    
    def run_analysis(self, query: str, maintain_context: bool = True) -> Dict:
        """
        Run complete analysis while optionally maintaining context.
        
        Args:
            query: The query to analyze
            maintain_context: Whether to maintain context for future queries
        """
        start_time = time.time()
        self.log(f"Starting analysis: {query}")
        
        # Clear context if not maintaining it
        if not maintain_context:
            self.context = OrchestratorContext()
        
        try:
            # Get execution plan
            plan = self._plan_execution(query)
            actors_used = [actor for actor, _ in plan]
            results = {}
            
            # Execute plan
            for actor_name, task in plan:
                result = self._execute_actor(actor_name, task)
                results[actor_name] = result["content"]
                self.log(f"Completed {actor_name}")
            
            # Create final synthesis prompt
            synthesis_prompt = f"""Query: {query}

Actor Results:
{chr(10).join(f'From {name}:\n{content}\n' for name, content in results.items())}

Previous Context Available:
{chr(10).join(f'- {actor}: {len(history)} previous results' for actor, history in self.context.actor_results.items())}

Create a final answer that:
1. Directly addresses the query
2. Integrates all actor insights
3. Maintains continuity with previous analysis
4. Is clear and concise"""

            # Use planner for final synthesis
            synthesis = self.planner.run_analysis(synthesis_prompt)
            execution_time = time.time() - start_time
            
            response = {
                "answer": synthesis["content"],
                "execution_time": execution_time,
                "steps_executed": len(plan),
                "actors_used": actors_used,
                "results": results
            }
            
            self.log(f"Analysis completed in {execution_time:.2f} seconds")
            return response
            
        except Exception as e:
            self.log(f"Error during execution: {str(e)}", "ERROR")
            raise
        
    def cleanup(self, full: bool = True):
        """
        Clean up resources.
        
        Args:
            full: If True, clears all context. If False, maintains thread IDs.
        """
        self.log("Starting cleanup...")
        
        try:
            # Always cleanup actors
            for agent, _ in self.actors.values():
                agent.cleanup()
            
            # Cleanup planner
            if hasattr(self, 'planner'):
                self.planner.cleanup()
            
            # Optionally clear context
            if full:
                self.context = OrchestratorContext()
            
            self.log("Cleanup completed")
            
        except Exception as e:
            self.log(f"Error during cleanup: {str(e)}", "ERROR")
            raise