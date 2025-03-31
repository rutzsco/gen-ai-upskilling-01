# Lab 01: Building an AI Agent with Function Calling in Semantic Kernel

## Overview

In this advanced lab, we'll extend our RAG application to implement the "agentic RAG" pattern using OpenAI function calling. Instead of always retrieving information for every query, we'll let the LLM decide when to use a retrieval tool based on the query content.

This approach makes our application more intelligent and efficient by letting the AI determine when to search for information rather than doing it for every user request.

## What You'll Learn

- How function calling works in Semantic Kernel
- The difference between static RAG and agentic RAG
- How to implement a retrieval plugin
- How to register and use plugins with Semantic Kernel
- How to create an AI agent that dynamically decides when to search

## Prerequisites

- Completion of Lab 03 (FastAPI RAG Application)
- Understanding of basic RAG patterns
- Familiarity with Semantic Kernel concepts

## Key Concepts

### Function Calling vs. Static RAG

Before diving into the code, let's understand the key differences between our previous RAG implementation and this new approach:

**Static RAG (Previous Lab):**
- Always retrieves information for every query
- Fixed processing pipeline: query → search → generate response
- Developer decides when to retrieve information

**Agentic RAG (This Lab):**
- LLM decides when to retrieve information
- Dynamic process: query → LLM determines if search is needed → conditionally search → generate response
- Model autonomously chooses when to use tools/functions

## Step 1: Creating the Retrieval Plugin

First, let's create a plugin that allows the LLM to search for information when needed. Create a new file called `retrival_plugins.py` in your `app/services` directory:

```python
from typing import Annotated
from dataclasses import dataclass
from typing import Annotated
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.kernel import Kernel
from semantic_kernel.functions.kernel_arguments import KernelArguments
from dataclasses import dataclass
from typing import Annotated
import os
from dataclasses import dataclass
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI

@dataclass
class KnowledgeSource:
    name: str
    content: str

class RetrivalPlugin:
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self.embeddings_client = AzureOpenAI(azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
                                             azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                                             api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                                             api_version="2024-10-21")
  
    @kernel_function(name="get_sources", description="Get relevant source information based on the search query")
    async def get_sources(self, arguments: KernelArguments, query_text: Annotated[str, "search query"]) -> Annotated[str, "The output is a string"]:
        
        try:
            # Get the embedding for the query text using Azure OpenAI
            query_embeddings = self.embeddings_client.embeddings.create(input=query_text, model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"), dimensions=3072).data[0].embedding

            # Search for knowledge articles using the Azure AI Search service
            search_client = SearchClient(endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"), index_name=os.getenv("AZURE_AI_SEARCH_INDEX_NAME"),credential=AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_API_KEY")))
            vector_query = VectorizedQuery(vector=query_embeddings, k_nearest_neighbors=5, fields="text_vector")
            results = search_client.search(search_text=None, vector_queries=[vector_query], select=["title", "chunk"], top=10)

            # Map the results to the KnowledgeSource dataclass
            knowledge_sources = [
                KnowledgeSource(name=result["title"], content=result["chunk"])
               for result in results
            ]

            return knowledge_sources
    
        except Exception as e:
            # Handle any errors that occur during the search
            print(f"An error occurred during the search: {e}")
            return []  # Return an empty list to indicate no results due to the error
```

**What's happening here:**

1. We've created a `RetrivalPlugin` class with a `get_sources` method decorated as a `kernel_function`
2. This function accepts a search query and returns relevant information from our knowledge base
3. The `@kernel_function` decorator exposes this method to the LLM as a callable function with a description
4. When the LLM determines it needs to search for information, it can invoke this function

## Step 2: Adding the Agent System Prompt

Create a new system prompt for the agent in `app/prompt/RAGAgentSystemPrompt.txt`:

```plaintext
**System Prompt for AutoAssist:**

You are AutoAssist, an intelligent assistant specialized in diagnosing and troubleshooting automotive issues. Your responses must be based solely on the content provided in the owner's manuals and related sources. A tool called **get_sources** is available for you to search the owner's manual using a provided search term; use this tool to retrieve relevant information.

Follow these guidelines:

1. **Source-Driven Responses:**  
   - Base all guidance exclusively on information from the provided sources.  
   - Every fact or piece of guidance must include a citation in square brackets indicating the source name (e.g., [source1.txt]).  
   - If multiple sources are used, list each citation separately (e.g., [source1.txt][source2.pdf]).  
   - Do not combine or merge information from different sources unless explicitly instructed.

2. **Utilizing the get_sources Tool:**  
   - When a query requires specific information from the owner's manual, use the **get_sources** tool with an appropriate search term to locate the necessary content.  
   - Only incorporate information from the results provided by **get_sources**.

3. **Detailed but Concise Answers:**  
   - Provide detailed troubleshooting steps and diagnostic guidance while remaining concise and directly addressing the user's question.

4. **Clarification Requests:**  
   - If you cannot answer a question using the available sources, politely ask the user for more details or clarification.

5. **Language and Tone:**  
   - Respond in the same language as the user's query if it is not in English.  
   - Maintain a professional, knowledgeable, and helpful tone at all times.

6. **Follow-Up Questions:**  
   - At the end of each response, generate three very brief follow-up questions that the user might ask next.  
   - Enclose these follow-up questions in double angle brackets, ensuring that the final question ends with ">>".  
   - Do not repeat questions that have already been asked.

7. **Source Listing:**  
   - At the end of your response, list all sources used.

By following these guidelines, you will provide accurate, source-based automotive support that leverages the owner's manuals effectively.
```

**What's different:**
- This prompt explicitly mentions the `get_sources` tool
- It guides the LLM on when and how to use the tool
- It emphasizes that information should come only from retrieved sources

## Step 3: Implementing the Agent Method in SemanticKernelService

Now, let's add a new method to our `SemanticKernelService` class in `app/services/sk.py` to handle the agent-based approach:

```python
async def run_rag_agent(self, request: ChatRequest) -> str:
    if not request.messages:
        raise ValueError("No messages found in request.")

    chat_completion_service = self.kernel.get_service(service_id="azure-chat-completion")
    
    settings=PromptExecutionSettings(
        function_choice_behavior=FunctionChoiceBehavior.Auto(filters={"included_plugins": ["rag_retrival"]}),
    )

    kernel_arguments = KernelArguments()
    
    system_message = self.file_service.read_file('RAGAgentSystemPrompt.txt')
    chat_history_1 = ChatHistory()
    chat_history_1.add_system_message(system_message)
    for message in request.messages:
        if message.role.lower() == "user":
            chat_history_1.add_user_message(message.content)
        elif message.role.lower() == "assistant":
            chat_history_1.add_assistant_message(message.content)

    
    chat_result = await chat_completion_service.get_chat_message_content(
        chat_history=chat_history_1,
          kernel_arguments=kernel_arguments, 
          settings=settings, 
          kernel=self.kernel)    
   
    kernel_arguments ["diagnostics"] = []
    request_result = RequestResult(
        content=f"{chat_result}",
        execution_diagnostics=ExecutionDiagnostics(steps=kernel_arguments ["diagnostics"])
    )

    return request_result
```

**Key differences from the standard RAG method:**

1. We use `PromptExecutionSettings` with `FunctionChoiceBehavior.Auto` to enable function calling
2. We specify which plugins can be used through the `included_plugins` filter
3. We pass the `kernel` object directly in the `get_chat_message_content` call to enable function execution
4. We don't manually perform the embedding/search steps - the model decides if and when to do this

## Step 4: Initialize and Register the Plugin

Also in `sk.py`, add code to register the plugin in the `__init__` method:

```python
def __init__(self):
    # ...existing initialization code...
    
    self.kernel.add_plugin(RetrivalPlugin(self.kernel), plugin_name="rag_retrival")
```

This registers our retrieval plugin with the kernel under the name "rag_retrival".

## Step 5: Add a New API Endpoint

Add a new endpoint to `app/routes/workflow.py`:

```python
@router.post("/rag-agent")
async def run_rag_agent(input_data: ChatRequest):
    """
    POST endpoint for executing a rag agent.
    """
    result = await sk_service.run_rag_agent(input_data)
    return {"result": result}
```

## Step 6: Testing the Agent

Create a test HTTP request in your HTTP client:

```http
POST http://127.0.0.1:8000/rag-agent
Content-Type: application/json

{
  "messages": [
    {
      "role": "User",
      "content": "What is the part number for the oil filter?"
    }
  ]
}
```

## Understanding How It Works

Let's break down what happens when this endpoint is called:

1. The user question "What is the part number for the oil filter?" is sent to the agent endpoint
2. The LLM receives the question along with the system prompt that describes the available tool
3. The model recognizes this query requires looking up information from the manual
4. It decides to call the `get_sources` function with an appropriate search term
5. The function executes the search and returns relevant sources
6. The LLM uses these sources to craft a response with proper citations
7. The response is returned to the user

This "agentic" approach has several advantages:
- The model can decide when information retrieval is necessary
- It can formulate better search queries based on the conversation context
- It can choose not to search when the query doesn't require external information
- It can interact with multiple tools if needed

## The Power of Function Calling

Function calling is a powerful capability that allows LLMs to:

1. **Recognize when to use tools**: The model analyzes the user's query and determines if external data or computation is needed
2. **Select the appropriate function**: When multiple functions are available, it chooses the most relevant one
3. **Format parameters correctly**: It structures the function parameters in the expected format
4. **Process the results**: It incorporates the function's output into its response

By implementing function calling, we've transformed our RAG system into a more intelligent agent that can dynamically decide how to solve problems.

## Static RAG vs. Agentic RAG: A Comparison

| Feature | Static RAG (Previous Lab) | Agentic RAG (This Lab) |
|---------|---------------------------|------------------------|
| Information retrieval | Always performed for every query | Performed only when the LLM deems necessary |
| Control flow | Predetermined by developer | Determined dynamically by the LLM |
| System architecture | Pipeline of sequential steps | Tool-using agent with decision-making ability |
| Flexibility | Limited to predetermined process | Can adapt approach based on query type |
| Efficiency | May retrieve information unnecessarily | Retrieves information only when needed |

## Conclusion

In this lab, you've learned how to implement function calling with Semantic Kernel to create an AI agent that can dynamically decide when to search for information. This approach represents a significant advancement over the static RAG pattern by allowing the LLM to determine its own problem-solving strategy.

As you continue to explore AI application development, consider how this pattern can be extended with additional tools and functions to create even more capable and autonomous agents.

