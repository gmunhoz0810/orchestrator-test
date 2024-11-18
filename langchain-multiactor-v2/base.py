from typing import Dict, List, Optional
from pathlib import Path
import os
from azure_config import get_azure_client

class CodeInterpreterAgent:
    """Code Interpreter Agent using Azure OpenAI"""
    
    # Initialize Azure OpenAI client at class level
    client = get_azure_client()
    
    def __init__(
        self,
        name: str,
        instructions: str,
        model: str = "a1sandboxcp4o",  # Azure deployment name
        temperature: float = 0.7,
        tools: Optional[List[Dict[str, str]]] = None,
        files: Optional[List[str]] = None,
        verbose: bool = True
    ):
        """Initialize the CodeInterpreterAgent."""
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
            print(f"\nInitializing with configuration:")
            print(f"Name: {self.name}")
            print(f"Model: {self.model}")
            print(f"Temperature: {self.temperature}")
            print(f"Tools: {self.tools}")
            print(f"Number of files: {len(self.files)}")

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
        """Initialize the Azure OpenAI assistant"""
        if self.verbose:
            print("\nStarting initialization...")
            
        # Upload files first
        file_ids = []
        for file_path in self.files:
            file_id = self._upload_file(file_path)
            file_ids.append(file_id)
            
        # Create assistant with instructions about files
        enhanced_instructions = f"{self.instructions}\n\n{self._create_file_instructions()}"
        
        try:
            # Create the assistant using Azure OpenAI
            assistant = self.client.beta.assistants.create(
                name=self.name,
                instructions=enhanced_instructions,
                tools=self.tools,
                model=self.model,
                file_ids=file_ids
            )
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

    def _upload_file(self, file_path: str) -> str:
        """Upload a file and maintain filename mapping."""
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
            
    def run_analysis(self, query: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Run analysis using the Azure OpenAI assistant."""
        if not self.assistant_id:
            self.initialize()
            
        # Create or use thread
        if not thread_id:
            thread = self.client.beta.threads.create()
            thread_id = thread.id
            
        # Add message to thread
        self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=query
        )
        
        # Run the assistant
        run = self.client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=self.assistant_id
        )
        
        # Wait for completion
        while True:
            run_status = self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            
            if run_status.status == "completed":
                break
            elif run_status.status in ["failed", "expired", "cancelled"]:
                raise Exception(f"Run failed with status: {run_status.status}")
                
        # Get messages
        messages = self.client.beta.threads.messages.list(thread_id=thread_id)
        
        # Get last assistant message
        for msg in messages.data:
            if msg.role == "assistant":
                content_value = None
                for content in msg.content:
                    if hasattr(content, 'text'):
                        content_value = content.text.value
                        break
                
                return {
                    "content": content_value or "No text content found",
                    "thread_id": thread_id,
                    "run_id": run.id
                }
                
        return {"error": "No assistant response found"}
    
    def cleanup(self):
        """Clean up resources."""
        for file_id in self.file_mapping:
            try:
                self.client.files.delete(file_id)
                if self.verbose:
                    print(f"Deleted file {file_id}")
            except Exception as e:
                print(f"Error deleting file {file_id}: {str(e)}")
        self.file_mapping.clear()