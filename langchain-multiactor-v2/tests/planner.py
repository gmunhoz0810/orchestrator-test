from typing import Dict, List, Tuple
from datetime import datetime
from metadata_models import (
    ExecutionPlan, ExecutionStep, ExecutionMetrics,
    DataType, AnalysisType, ActorCapabilities
)
from openai import OpenAI
import json
import re

class QueryPlanner:
    """
    Analyzes queries and creates efficient execution plans.
    """
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.actor_capabilities: Dict[str, ActorCapabilities] = {}
        self.debug = True
        
    def register_actor(self, name: str, capabilities: ActorCapabilities) -> None:
        """Register an actor and its capabilities"""
        self.actor_capabilities[name] = capabilities
        
    def _clean_json_response(self, response: str) -> str:
        """Clean the response to get pure JSON"""
        # Remove markdown code block if present
        if '```' in response:
            # Extract content between ```json and ```
            match = re.search(r'```(?:json)?(.*?)```', response, re.DOTALL)
            if match:
                response = match.group(1)
        # Remove any leading/trailing whitespace
        return response.strip()
        
    def _analyze_query_requirements(self, query: str) -> Tuple[List[DataType], List[AnalysisType]]:
        """
        Analyze what data types and analyses are needed for a query.
        Uses LLM to understand query requirements.
        """
        capabilities_desc = self._format_capabilities_for_prompt()
        
        data_types = [dt.name for dt in DataType]
        analysis_types = [at.name for at in AnalysisType]
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that analyzes queries to determine required data types and analyses. Respond with a JSON object containing required_data_types and required_analyses arrays."
            },
            {
                "role": "user",
                "content": f"""Analyze this query: "{query}"

Available capabilities:
{capabilities_desc}

Valid data types: {data_types}
Valid analysis types: {analysis_types}

Return a JSON object with this structure:
{{"required_data_types": ["CUSTOMER"], "required_analyses": ["GEOGRAPHIC"]}}

Use only the valid types listed above."""
            }
        ]

        if self.debug:
            print("\nSending request to OpenAI with messages:")
            print(json.dumps(messages, indent=2))
            
        response = self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=messages,
            temperature=0
        )
        
        if self.debug:
            print("\nReceived response from OpenAI:")
            print(response.choices[0].message.content)
            
        try:
            # Clean the response content
            cleaned_content = self._clean_json_response(response.choices[0].message.content)
            
            if self.debug:
                print("\nCleaned JSON content:")
                print(cleaned_content)
            
            result = json.loads(cleaned_content)
            
            # Validate response format
            if not isinstance(result, dict):
                raise ValueError("Response is not a dictionary")
            if "required_data_types" not in result or "required_analyses" not in result:
                raise ValueError("Response missing required keys")
            if not all(dt in data_types for dt in result["required_data_types"]):
                raise ValueError("Invalid data types in response")
            if not all(at in analysis_types for at in result["required_analyses"]):
                raise ValueError("Invalid analysis types in response")
            
            return (
                [DataType[dt] for dt in result["required_data_types"]],
                [AnalysisType[at] for at in result["required_analyses"]]
            )
            
        except json.JSONDecodeError as e:
            print(f"\nJSON Parse Error: {e}")
            print("Response content:", response.choices[0].message.content)
            print("Cleaned content:", cleaned_content)
            raise ValueError(f"Failed to parse LLM response: {e}")
        except Exception as e:
            print(f"\nValidation Error: {e}")
            print("Parsed result:", result)
            raise ValueError(f"Invalid response format: {e}")

    def _format_capabilities_for_prompt(self) -> str:
        """Format all actor capabilities into a string for prompts"""
        output = []
        for actor_name, caps in self.actor_capabilities.items():
            output.append(f"Actor: {actor_name}")
            output.append("Data Types:")
            for dt in caps.supported_data_types:
                output.append(f"- {dt.name}")
            output.append("Analyses:")
            for at in caps.supported_analyses:
                output.append(f"- {at.name}")
            output.append("")
        return "\n".join(output)
    
        
    def _create_execution_steps(
        self,
        required_data: List[DataType],
        required_analyses: List[AnalysisType]
    ) -> List[ExecutionStep]:
        """Create ordered list of execution steps"""
        steps = []
        handled_data = set()
        
        # First, get all required data
        for data_type in required_data:
            # Find actor that can handle this data
            actor = self._find_best_actor_for_data(data_type)
            step = ExecutionStep(
                actor_name=actor,
                action=f"retrieve_{data_type.name.lower()}_data",
                required_data=[data_type],
                expected_output=f"{data_type.name} data summary"
            )
            steps.append(step)
            handled_data.add(data_type)
            
        # Then, handle analyses
        for analysis in required_analyses:
            # Find actor that can do this analysis
            actor = self._find_best_actor_for_analysis(analysis)
            dependencies = [
                s.actor_name for s in steps
                if any(dt in s.required_data for dt in handled_data)
            ]
            step = ExecutionStep(
                actor_name=actor,
                action=f"perform_{analysis.name.lower()}_analysis",
                required_data=list(handled_data),
                expected_output=f"{analysis.name} analysis results",
                depends_on=dependencies
            )
            steps.append(step)
            
        return steps
        
    def _find_best_actor_for_data(self, data_type: DataType) -> str:
        """Find the most suitable actor for handling a data type"""
        for name, caps in self.actor_capabilities.items():
            if data_type in caps.supported_data_types:
                return name
        raise ValueError(f"No actor found for data type: {data_type}")
        
    def _find_best_actor_for_analysis(self, analysis_type: AnalysisType) -> str:
        """Find the most suitable actor for an analysis type"""
        for name, caps in self.actor_capabilities.items():
            if analysis_type in caps.supported_analyses:
                return name
        raise ValueError(f"No actor found for analysis: {analysis_type}")
        
    def create_execution_plan(self, query: str) -> ExecutionPlan:
        """
        Create a complete execution plan for a query.
        """
        # Start tracking metrics
        metrics = ExecutionMetrics(start_time=datetime.now())
        
        # Analyze query requirements
        required_data, required_analyses = self._analyze_query_requirements(query)
        
        # Create execution steps
        steps = self._create_execution_steps(required_data, required_analyses)
        
        # Create synthesis prompt
        synthesis_prompt = self._create_synthesis_prompt(query, steps)
        
        # Get unique actors needed
        required_actors = list(set(step.actor_name for step in steps))
        
        return ExecutionPlan(
            query=query,
            steps=steps,
            final_synthesis_prompt=synthesis_prompt,
            required_actors=required_actors,
            metrics=metrics
        )
        
    def _create_synthesis_prompt(self, query: str, steps: List[ExecutionStep]) -> str:
        """Create the prompt for final synthesis of results"""
        return f"""Original query: {query}
        
        After collecting analyses from different actors:
        {', '.join(s.expected_output for s in steps)}
        
        Please synthesize a clear, concise answer that:
        1. Directly addresses the original query
        2. Integrates insights from all analyses
        3. Maintains clarity and brevity
        4. Avoids repetition
        5. Highlights key findings first
        """