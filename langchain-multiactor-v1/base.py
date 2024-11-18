from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from openai import OpenAI
import os
from dotenv import load_dotenv
import time

# Load environment variables at module level
load_dotenv()

class CodeInterpreterAgent:
    """
    A LangChain agent that uses OpenAI's Assistant API with file management capabilities
    that preserve original filenames.
    """
    
    # Initialize OpenAI client at class level
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def __init__(
        self,
        name: str,
        instructions: str,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        tools: Optional[List[Dict[str, str]]] = None,
        files: Optional[List[str]] = None,
        verbose: bool = True
    ):
        """Initialize the CodeInterpreterAgent."""
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        self.name = name
        self.instructions = instructions
        self.model = model
        self.temperature = temperature
        self.verbose = verbose
        
        # Set default tools if none provided
        self.tools = tools or [{"type": "code_interpreter"}]
        
        # Initialize file management
        self.file_mapping: Dict[str, Dict[str, str]] = {}
        self.files = files or []
        self.assistant_id = None
        
        if self.verbose:
            print("\nInitializing with configuration:")
            print(f"Name: {self.name}")
            print(f"Model: {self.model}")
            print(f"Temperature: {self.temperature}")
            print(f"Tools: {self.tools}")
            print(f"Number of files: {len(self.files)}")

    def _upload_file(self, file_path: str) -> str:
        """Upload a file to OpenAI and maintain filename mapping."""
        if self.verbose:
            print(f"\nAttempting to upload file: {file_path}")
            
        try:
            with open(file_path, 'rb') as file:
                response = self.client.files.create(
                    file=file,
                    purpose='assistants'
                )
                
                self.file_mapping[response.id] = {
                    'filename': Path(file_path).name,
                    'path': file_path
                }
                
                if self.verbose:
                    print(f"Successfully uploaded {Path(file_path).name} → file ID: {response.id}")
                
                return response.id
        except Exception as e:
            print(f"Error uploading file {file_path}: {str(e)}")
            raise

    def _create_file_instructions(self) -> str:
        """Create instructions about file mappings."""
        if not self.file_mapping:
            return ""
            
        file_info = "\n".join([
            f"- File ID '{file_id}' is '{info['filename']}'"
            for file_id, info in self.file_mapping.items()
        ])
        
        return f"""
IMPORTANT - File name mapping information:
When working with files, please note the following filename mappings:
{file_info}

When reading files in your code, use the File IDs, but refer to the original filenames in your communications.
"""

    def initialize(self) -> None:
        """Initialize the OpenAI assistant with file management."""
        if self.verbose:
            print("\nStarting initialization...")
            
        # Upload all files first and collect their IDs
        file_ids = []
        for file_path in self.files:
            file_id = self._upload_file(file_path)
            file_ids.append(file_id)
            
        # Enhance instructions with file mapping information
        enhanced_instructions = f"{self.instructions}\n\n{self._create_file_instructions()}"
        
        if self.verbose:
            print("\nCreating assistant with:")
            print(f"Enhanced instructions: {enhanced_instructions[:200]}...")
            print(f"File IDs: {file_ids}")
        
        try:
            # Create the assistant with uploaded files
            creation_params = {
                "name": self.name,
                "instructions": enhanced_instructions,
                "tools": self.tools,
                "model": self.model,
            }

            # Only add tool_resources if we have files
            if file_ids:
                creation_params["tool_resources"] = {
                    "code_interpreter": {
                        "file_ids": file_ids
                    }
                }
                
            if self.verbose:
                print("\nCreation parameters:")
                print(creation_params)

            assistant = self.client.beta.assistants.create(**creation_params)
            self.assistant_id = assistant.id
            
            if self.verbose:
                print("\nAssistant created successfully:")
                print(f"Assistant ID: {self.assistant_id}")
                print(f"Model: {self.model}")
                print(f"Tools: {', '.join(t['type'] for t in self.tools)}")
                print(f"Files attached: {len(file_ids)}")
                
        except Exception as e:
            print(f"\nError creating assistant: {str(e)}")
            raise

    def run_analysis(
        self, 
        query: str,
        additional_files: Optional[List[str]] = None,
        thread_id: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: int = 5
    ) -> Dict[str, Any]:
        """Run analysis using the assistant."""
        if not self.assistant_id:
            self.initialize()
            
        if self.verbose:
            print(f"\nStarting analysis with query: {query}")
            
        # Create a new thread if none exists
        if not thread_id:
            thread = self.client.beta.threads.create()
            thread_id = thread.id
            if self.verbose:
                print(f"Created new thread: {thread_id}")
                
        # Create the message
        message = self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=query
        )
        
        if self.verbose:
            print(f"Created message in thread {thread_id}")
        
        # Create and run the assistant
        run = self.client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=self.assistant_id
        )
        
        if self.verbose:
            print(f"Started run {run.id}")
        
        # Wait for completion with retries
        retries = 0
        while True:
            try:
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id
                )
                
                if self.verbose:
                    print(f"Run status: {run.status}")
                
                if run.status == "completed":
                    break
                elif run.status in ["failed", "expired", "cancelled"]:
                    raise Exception(f"Run failed with status: {run.status}")
                elif retries >= max_retries:
                    raise Exception("Max retries reached")
                    
                time.sleep(1)  # Short delay between status checks
                
            except Exception as e:
                retries += 1
                if retries == max_retries:
                    raise Exception(f"Max retries reached: {str(e)}")
                print(f"Retry {retries}/{max_retries} after error: {str(e)}")
                time.sleep(retry_delay)
        
        # Get the messages
        messages = self.client.beta.threads.messages.list(
            thread_id=thread_id
        )
        
        if self.verbose:
            print("Retrieved messages")
        
        # Return the last assistant message
        for msg in messages.data:
            if msg.role == "assistant":
                # Handle different content types
                content_parts = []
                for content in msg.content:
                    if hasattr(content, 'text'):
                        content_parts.append(content.text.value)
                    elif hasattr(content, 'image_file'):
                        content_parts.append("[Image generated]")
                    else:
                        content_parts.append("[Content of unsupported type]")
                
                response_content = "\n".join(content_parts)
                return {
                    "content": response_content,
                    "thread_id": thread_id,
                    "run_id": run.id
                }
            
        return {"error": "No assistant response found"}

    def cleanup(self):
        """Clean up by deleting uploaded files."""
        if self.verbose:
            print("\nStarting cleanup...")
            
        for file_id in self.file_mapping:
            try:
                self.client.files.delete(file_id)
                if self.verbose:
                    print(f"Deleted file {file_id}")
            except Exception as e:
                print(f"Error deleting file {file_id}: {str(e)}")
        self.file_mapping.clear()
        
        if self.verbose:
            print("Cleanup completed")