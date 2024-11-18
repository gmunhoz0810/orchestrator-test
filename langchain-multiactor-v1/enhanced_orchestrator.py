from typing import Dict, List, Tuple
from base import CodeInterpreterAgent
from actor_metadata import ActorMetadata
import time
from openai import OpenAI
import json

class OrchestratorContext:
    """Maintains context between runs"""
    def __init__(self):
        self.threads: Dict[str, str] = {}
        self.actor_results: Dict[str, List[Dict]] = {}
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
    def __init__(self, api_key: str, verbose: bool = True):
        self.client = OpenAI(api_key=api_key)
        self.actors: Dict[str, Tuple[CodeInterpreterAgent, ActorMetadata]] = {}
        self.verbose = verbose
        self.context = OrchestratorContext()
        
    def log(self, message: str, level: str = "INFO"):
        if self.verbose:
            print(f"\n[Orchestrator:{level}] {message}")
    
    def register_actor(self, name: str, agent: CodeInterpreterAgent, metadata: ActorMetadata):
        self.actors[name] = (agent, metadata)
        self.log(f"Registered {name} with access to {metadata.file_path}")
    
    def _escape_backslashes(self, value: str) -> str:
        return value.replace("\\", "\\\\")
    
    def _create_actor_prompt(self, actor_name: str, task: str) -> str:
        history = self._escape_backslashes(self.context.get_actor_history(actor_name))
        task = self._escape_backslashes(task)
        
        prompt = f"""Task: {task}

Previous Context:
{history}

Requirements:
1. Use the file ID provided in your initialization.
2. Reference previous analysis if relevant.
3. Be specific and quantitative.
4. Return clear, structured results."""
        return prompt
    
    def _plan_execution(self, query: str) -> List[Tuple[str, str]]:
        actor_descriptions = []
        for name, (_, meta) in self.actors.items():
            desc = f"Actor '{name}':\n- Data: {meta.data_description}\n- Previous context: {'Yes' if name in self.context.actor_results else 'No'}\n"
            actor_descriptions.append(desc)
        
        system_prompt = """Create an efficient execution plan with one call per actor, minimizing intermediate steps.
        1. Consolidate tasks where possible to reduce actor interactions.
        2. Leverage existing context from previous runs.
        3. Ensure continuity of analysis without unnecessary redundancies."""

        user_prompt = f"""Query: "{query}"

Available actors:
{chr(10).join(actor_descriptions)}

Create a minimal execution plan. Format:
1. actor_name: complete_task_description"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        
        steps = []
        for line in response.choices[0].message.content.strip().split("\n"):
            if ':' in line and line.strip():
                actor, task = line.split(":", 1)
                actor = actor.strip().split(".", 1)[-1].strip()
                if actor in self.actors:
                    steps.append((actor, self._escape_backslashes(task.strip())))
        
        self.log("Execution plan:")
        for i, (actor, task) in enumerate(steps, 1):
            self.log(f"{i}. {actor} -> {task}")
            
        return steps
    
    def _execute_actor(self, actor_name: str, task: str) -> Dict:
        agent, _ = self.actors[actor_name]
        self.log(f"Executing {actor_name} with task: {task}")
        prompt = self._create_actor_prompt(actor_name, task)
        thread_id = self.context.threads.get(actor_name)
        result = agent.run_analysis(prompt, thread_id=thread_id)
        if not thread_id and result.get("thread_id"):
            self.context.threads[actor_name] = result["thread_id"]
            self.log(f"Created thread for {actor_name}: {result['thread_id']}")
        self.context.add_result(actor_name, result)
        return result
    
    def run_analysis(self, query: str, maintain_context: bool = True) -> Dict:
        start_time = time.time()
        self.log(f"Starting analysis: {query}")
        if not maintain_context:
            self.context = OrchestratorContext()
        plan = self._plan_execution(query)
        results = {}
        for actor_name, task in plan:
            result = self._execute_actor(actor_name, task)
            results[actor_name] = result["content"]
        
        execution_time = time.time() - start_time
        final_answer = "\n".join(f"From {actor}:\n{response}" for actor, response in results.items())
        return {
            "answer": final_answer,
            "execution_time": execution_time,
            "steps_executed": len(plan),
            "actors_used": [actor for actor, _ in plan],
            "results": results,
        }
    
    def cleanup(self, full: bool = True):
        self.log("Starting cleanup...")
        for agent, _ in self.actors.values():
            agent.cleanup()
        if full:
            self.context = OrchestratorContext()
        self.log("Cleanup completed")
