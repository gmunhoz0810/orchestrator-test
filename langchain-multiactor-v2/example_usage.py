from enhanced_orchestrator import EnhancedOrchestrator
from actor_metadata import CUSTOMER_ACTOR_METADATA, ORGANIZATION_ACTOR_METADATA
from base import CodeInterpreterAgent
import os
from dotenv import load_dotenv

def main():
    # Load environment
    load_dotenv()
    
    # Initialize orchestrator (no API key needed, it's handled in azure_config)
    orchestrator = EnhancedOrchestrator(verbose=True)
    
    # Create and register actors
    customer_agent = CodeInterpreterAgent(
        name=CUSTOMER_ACTOR_METADATA.name,
        instructions=CUSTOMER_ACTOR_METADATA.system_prompt,
        files=[CUSTOMER_ACTOR_METADATA.file_path],
        model="a1sandboxcp4o"  # Azure deployment name
    )

    org_agent = CodeInterpreterAgent(
        name=ORGANIZATION_ACTOR_METADATA.name,
        instructions=ORGANIZATION_ACTOR_METADATA.system_prompt,
        files=[ORGANIZATION_ACTOR_METADATA.file_path],
        model="a1sandboxcp4o"  # Azure deployment name
    )
    
    # Register actors
    orchestrator.register_actor("customer_specialist", customer_agent, CUSTOMER_ACTOR_METADATA)
    orchestrator.register_actor("organization_specialist", org_agent, ORGANIZATION_ACTOR_METADATA)
    
    try:
        # Initial query
        query1 = "For the top 3 countries by number of customers, what are the most common industries?"
        result1 = orchestrator.run_analysis(query1, maintain_context=True)
        
        print("\nFirst Query Results:")
        print(f"Execution time: {result1['execution_time']:.2f} seconds")
        print(f"Steps executed: {result1['steps_executed']}")
        print(f"Actors used: {', '.join(result1['actors_used'])}")
        print("\nDetailed responses:")
        for actor, response in result1['results'].items():
            print(f"\n{actor.upper()}:")
            print(response)
        print("\nFinal Answer:")
        print(result1['answer'])
        
    finally:
        # Only do full cleanup at the very end
        orchestrator.cleanup(full=True)

if __name__ == "__main__":
    main()