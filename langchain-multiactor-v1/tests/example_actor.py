from base import CodeInterpreterAgent

INSTRUCTIONS = """You are a data analyst who helps analyze CSV files. 
Always explain your analysis steps and refer to files by their original names in your explanations.
When analyzing data:
1. First, examine the structure and contents of the data
2. Provide clear explanations for each analysis step
3. Include relevant statistical measures and visualizations
4. Highlight any significant patterns or anomalies
5. Present conclusions in business-friendly language"""

def main():
    # Initialize the agent with files
    agent = CodeInterpreterAgent(
        name="Data Analysis Assistant",
        instructions=INSTRUCTIONS,
        files=[
            "data/customers-10000.csv",
            "data/organizations-10000.csv"
        ],
        verbose=True
    )

    try:
        # Run an analysis
        response = agent.run_analysis(
            "What are our file names to file ids mapping?"
        )
        print("\nInitial Analysis Response:")
        print(response.get("content", "No response content found"))

        # Example of following up with the same thread
        if response.get("thread_id"):
            follow_up = agent.run_analysis(
                "What are the top 5 industries in this dataset?",
                thread_id=response["thread_id"]
            )
            print("\nFollow-up Analysis Response:")
            print(follow_up.get("content", "No response content found"))

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    finally:
        # Clean up when done
        agent.cleanup()

if __name__ == "__main__":
    main()