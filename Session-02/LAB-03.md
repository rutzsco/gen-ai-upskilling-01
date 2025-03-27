# Lab 03: Building an AI-powered RAG Application with FastAPI and Semantic Kernel

This lab will guide you through creating a Retrieval-Augmented Generation (RAG) application using FastAPI and Microsoft's Semantic Kernel. By the end of this lab, you'll have a fully functional API that uses Azure OpenAI and Azure AI Search to implement a powerful RAG pattern.

## What You'll Learn

- How to set up a FastAPI application
- How to use Semantic Kernel to work with Large Language Models (LLMs)
- How to implement the RAG pattern with Azure AI Search
- How to structure your application for maintainability

## Prerequisites

- Python 3.11+
- Visual Studio Code
- An Azure account with access to Azure OpenAI and Azure AI Search
- Basic knowledge of Python

## Lab Structure

This lab is divided into the following sections:

1. Setting up the project environment
2. Creating the FastAPI application structure
3. Implementing the data models
4. Building the core services with Semantic Kernel
5. Creating the RAG workflow
6. Testing the application

## 1. Setting up the Project Environment

First, let's set up our project environment and install the necessary dependencies.

```bash
# Create a project directory
mkdir app
cd app

# Install required packages
pip install uvicorn fastapi pydantic python-dotenv requests semantic_kernel azure-search-documents azure-monitor-opentelemetry
```

Create a requirements.txt file to document our dependencies:

```python
uvicorn==0.34.0
fastapi==0.115.11
pydantic==2.10.6
python-dotenv==1.0.1
Requests==2.32.3
semantic_kernel==1.24.1
azure-search-documents==11.6.0b7
azure-monitor-opentelemetry==1.6.4
```

Now, let's create a `.env` file to store our environment variables:

```properties
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="gpt-4o"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME="text-embedding-3-large"
AZURE_OPENAI_ENDPOINT="YOUR_AZURE_OPENAI_ENDPOINT"
AZURE_OPENAI_API_KEY="YOUR_AZURE_OPENAI_API_KEY"

AZURE_AI_SEARCH_API_KEY="YOUR_SEARCH_API_KEY"
AZURE_AI_SEARCH_ENDPOINT="YOUR_SEARCH_ENDPOINT"
AZURE_AI_SEARCH_INDEX_NAME ="YOUR_INDEX_NAME"
```

## 2. Creating the FastAPI Application Structure

Let's create the basic structure of our FastAPI application. First, create the following directory structure:

```
demo-rag-api/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models/
│   │   └── api_models.py
│   ├── routes/
│   │   ├── status.py
│   │   └── workflow.py
│   ├── services/
│   │   └── sk.py
│   ├── prompt/
│   │   ├── file_service.py
│   │   ├── RAGSystemPrompt.txt
│   │   └── RAGSearchSystemPrompt.txt
│   └── config/
│       └── settings.py
├── .env
├── requirements.txt
└── README.md
```

## 3. Implementing the Data Models

Let's start by creating our data models in `app/models/api_models.py`:

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class ExecutionStep:
    name: str
    content: str

@dataclass
class ExecutionDiagnostics:
    steps: List[ExecutionStep] = field(default_factory=list)

@dataclass
class RequestResult:
    """A simple DTO for function call results."""
    content: str
    execution_diagnostics: ExecutionDiagnostics = field(default_factory=ExecutionDiagnostics)

@dataclass
class ChatMessage:
    role: str
    content: str
    user: str = None

@dataclass
class ChatRequest:
    messages: List[ChatMessage] = field(default_factory=list)
```

These models define the structure of our requests and responses. The `ChatRequest` contains a list of messages with roles (user or assistant) and content. The `RequestResult` will hold the response from our RAG system.

## 4. Setting Up FastAPI Routes

Now, let's create two simple routes: one for checking the status of our API and another for handling the RAG workflow.

First, create `app/routes/status.py`:

```python
from fastapi import APIRouter
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/status")
async def status():
    logger.info("Status Endpoint")
    return {"message": "Hello World"}
```

Next, create `app/routes/workflow.py`:

```python
from fastapi import APIRouter
from pydantic import BaseModel
from app.models.api_models import ChatRequest
from app.services.sk import SemanticKernelService
import asyncio

router = APIRouter()

sk_service = SemanticKernelService()

@router.post("/rag")
async def run_rag_workflow(input_data: ChatRequest):
    """
    POST endpoint for executing a RAG workflow.
    """
    result = await sk_service.run_rag(input_data)
    return {"result": result}
```

## 5. Creating the Prompt Service

Create a file service to manage our prompt templates in `app/prompt/file_service.py`:

```python
class FileService:
    """
    FileService is a class that provides functionality to manage and read files 
    using a mapping of file names to their full paths.
    """
    def __init__(self):
        # Initialize a dictionary to map file names to their full paths
        self.file_map = {}
        self.add_file('RAGSystemPrompt.txt', './app/prompt/RAGSystemPrompt.txt') 
        self.add_file('RAGSearchSystemPrompt.txt', './app/prompt/RAGSearchSystemPrompt.txt') 
        
    def add_file(self, file_name, file_path):
        self.file_map[file_name] = file_path

    def read_file(self, file_name):
        if file_name in self.file_map:
            file_path = self.file_map[file_name]
            try:
                with open(file_path, 'r') as file:
                    content = file.read()
                return content
            except FileNotFoundError:
                raise RuntimeError(f"File '{file_name}' not found at path '{file_path}'.")
        else:
            raise RuntimeError(f"File '{file_name}' not found in the file map.")
```

Now create the prompt templates:

For `app/prompt/RAGSystemPrompt.txt`:
```
You are an intelligent assistant helping professionals to diagnose and troubleshoot IT issues.
Suggest potential guidance using only the sources provided.
Be detailed but concise in your response.
Each source has a name followed by the actual information, always include the source name for each fact you use in the response.
If you cannot answer using the sources below ask the user for more information.
If an answer to the question is provided, it must be annotated with a citation. Use square brackets to reference the source, e.g. [info1.txt]. 
Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf]
List all sources used in the response at the end of the response.
If a user prompts in a non English language respond to them in their spoken language.
Generate 3 very brief follow-up questions that the user would likely ask next. 
Enclose the follow-up questions in double angle brackets. Example:
<<Are there exclusions for the policy?>>
<<Can you sumurrize the content in 3 points?>>
<<What can I look for the policy document?>>
Do no repeat questions that have already been asked.
Make sure the last question ends with ">>"
```

For `app/prompt/RAGSearchSystemPrompt.txt`:
```
Proivided is a history of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base.
You have access to Azure AI Search index with thousands of documents.
Generate a search query based on the conversation and the new question.
Do not include cited source filenames and document names e.g info.txt or doc.pdf in the search query terms.
Do not include any text inside [] or <<>> in the search query terms.
Do not include any special characters like '+'.
If the question is not in English, translate the question to English before generating the search query.
Only return the search query as the answer and less than 10 words total.
If you cannot generate a search query, return just the number 0.
```

## 6. Implementing the Semantic Kernel Service

Now let's create the Semantic Kernel service in `app/services/sk.py`, which is the heart of our RAG implementation:

```python
from semantic_kernel.contents import ChatHistory
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from dotenv import load_dotenv
import os
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from opentelemetry import trace
from app.models.api_models import ChatRequest, ExecutionDiagnostics, RequestResult
from app.prompt.file_service import FileService
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

class SemanticKernelService:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        # Retrieve configuration from environment variables
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        
        if not api_key or not endpoint or not deployment_name:
            raise ValueError("Missing required environment variables for OpenAI configuration.")
    
        # Initialize Semantic Kernel
        self.kernel = sk.Kernel()
        self.kernel.add_service(AzureChatCompletion(
            api_key=api_key,
            endpoint=endpoint,
            deployment_name=deployment_name, 
            service_id="azure-chat-completion"
        ))

        # Initialize FileService for prompt templates
        self.file_service = FileService()

        # Initialize the embeddings client
        self.embeddings_client = AzureOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-10-21"
        )

        # Initialize Azure AI Search client
        self.search_client = SearchClient(
            endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"), 
            index_name=os.getenv("VectorSearchIndexName_KnowledgeArticles"),
            credential=AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_API_KEY"))
        )

    async def run_rag(self, request: ChatRequest) -> str:      
        if not request.messages:
            raise ValueError("No messages found in request.")

        # Get chat completion service and execution settings
        chat_completion_service = self.kernel.get_service(service_id="azure-chat-completion")
        execution_settings = self.kernel.get_prompt_execution_settings_from_service_id("azure-chat-completion")

        # Step 1: Get the search query from the user conversation
        search_system_message = self.file_service.read_file('RAGSearchSystemPrompt.txt') 
        search_chat_history = ChatHistory()
        search_chat_history.add_system_message(search_system_message)
        
        # Add user messages to chat history
        for message in request.messages:
            if message.role.lower() == "user":
                search_chat_history.add_user_message(message.content)
            elif message.role.lower() == "assistant":
                search_chat_history.add_assistant_message(message.content)
                
        # Get search query from LLM
        search_chat_result = await chat_completion_service.get_chat_message_content(
            chat_history=search_chat_history, 
            settings=execution_settings
        )
        search_query = f"{search_chat_result}"
        
        # Step 2: Get the embeddings for the search query
        user_question = request.messages[-1].content
        query_embeddings = self.embeddings_client.embeddings.create(
            input=search_query, 
            model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
        ).data[0].embedding

        # Step 3: Perform a vector search in Azure AI Search
        vector_query = VectorizedQuery(
            vector=query_embeddings, 
            k_nearest_neighbors=5, 
            fields="embeddings"
        )
        search_results = self.search_client.search(
            search_text=None, 
            vector_queries=[vector_query], 
            select=["sourcefile", "content"], 
            top=10
        )
        
        # Format search results
        results = []
        for result in search_results:
            source = result.get("sourcefile", "Unknown Source")
            content = result.get("content", "No Content")
            results.append(f"<source><name>{source}</name><content>{content}</content></source>")
        results = "\n".join(results)

        # Step 4: Create a chat history and add the system message and user question
        system_message = self.file_service.read_file('RAGSystemPrompt.txt')
        chat_history_1 = ChatHistory()
        chat_history_1.add_system_message(system_message)
        
        # Add previous conversation context
        for message in request.messages[:-1]:
            if message.role.lower() == "user":
                chat_history_1.add_user_message(message.content)
            elif message.role.lower() == "assistant":
                chat_history_1.add_assistant_message(message.content)

        # Create user message with retrieved sources and the original question
        user_message = ("Sources:\n\n" + results + "\n\n" + "\n\nQuestion: " + user_question)
        chat_history_1.add_user_message(user_message)
        
        # Get response from LLM
        chat_result = await chat_completion_service.get_chat_message_content(
            chat_history=chat_history_1, 
            settings=execution_settings
        )
       
        # Create and return the response
        kernel_arguments = KernelArguments()
        kernel_arguments["diagnostics"] = []
        request_result = RequestResult(
            content=f"{chat_result}",
            execution_diagnostics=ExecutionDiagnostics(steps=kernel_arguments["diagnostics"])
        )

        return request_result
```

## 7. Creating the Main Entry Point

Now, let's create our main FastAPI application in `app/main.py`:

```python
from fastapi import FastAPI
from .routes.workflow import router as workflow_router
from .routes.status import router as status_router
import logging
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Instrumenting the requests library for OpenTelemetry tracing
RequestsInstrumentor().instrument()

# FastAPI app setup
app = FastAPI(title="RAG API with Semantic Kernel")
app.include_router(workflow_router)
app.include_router(status_router)
FastAPIInstrumentor.instrument_app(app)

# Basic config settings
@app.get("/")
async def root():
    return {"message": "Welcome to the RAG API with Semantic Kernel"}
```

## 8. Running and Testing the Application

Now that we've built all the necessary components, let's run our application:

```bash
uvicorn app.main:app --reload
```

This will start the FastAPI server with hot-reloading enabled. You can access the API documentation at http://127.0.0.1:8000/docs.

To test our RAG endpoint, you can use the following curl command or use the Swagger UI:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/rag' \
  -H 'Content-Type: application/json' \
  -d '{
  "messages": [
    {
      "role": "User",
      "content": "What is the part number for the oil filter?"
    }
  ]
}'
```

## Key Concepts Explained

### FastAPI
FastAPI is a modern, fast web framework for building APIs with Python. It's built on top of Starlette for the web parts and Pydantic for the data validation. In our application, FastAPI handles:
- Routing HTTP requests
- Request/response validation
- API documentation (OpenAPI)
- Dependency injection

### Semantic Kernel
Semantic Kernel is an open-source SDK from Microsoft that integrates Large Language Models (LLMs) with conventional programming languages. Key components we've used:
- **Kernel**: The central object that orchestrates everything
- **ChatCompletion**: The service for interacting with chat models
- **ChatHistory**: Manages conversation context and history
- **PromptExecutionSettings**: Controls model behavior during execution

### RAG Pattern
The Retrieval-Augmented Generation (RAG) pattern combines retrieval of external knowledge with LLM generation. Our implementation follows these steps:
1. Generate a search query from the user's question
2. Convert the query to vector embeddings
3. Search for relevant documents in Azure AI Search
4. Combine retrieved documents with the user's question
5. Generate a response from the LLM using the documents as context

This approach helps to:
- Ground the LLM with up-to-date information
- Reduce hallucinations
- Provide source citations for transparency

## Conclusion

Congratulations! You've successfully built a complete RAG application using FastAPI and Semantic Kernel. This application demonstrates how to:
- Set up a web API with FastAPI
- Integrate with Azure OpenAI using Semantic Kernel
- Implement the RAG pattern with Azure AI Search
- Structure your application in a maintainable way

You can extend this application by:
- Adding authentication
- Implementing more advanced RAG techniques
- Adding additional endpoints for different AI scenarios
- Improving error handling and logging
- Setting up monitoring and performance tracking
