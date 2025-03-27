# Lab 02: Enhancing RAG with Vector Search in Semantic Kernel

## Overview
In this advanced lab, you'll enhance the basic RAG application from our previous example by implementing vector search using Azure AI Search. Instead of using hardcoded documents, you'll leverage embeddings and vector search to retrieve the most relevant content for user questions.

## Prerequisites
- Completion of the basic RAG lab (lab_guide_sk_rag.md)
- Access to Azure AI Search
- Azure OpenAI account with access to embedding models
- Understanding of basic vector search concepts

## Setup

1. Install additional required packages:
   ```bash
   pip install openai azure-search-documents azure-core
   ```
   These packages enable interaction with Azure OpenAI embeddings and Azure AI Search services. The `openai` package lets us generate text embeddings, while `azure-search-documents` provides the client for vector search operations.

2. Update your `.env` file with additional credentials:
   ```
   # Existing credentials
   AZURE_OPENAI_API_KEY=your_api_key
   AZURE_OPENAI_ENDPOINT=your_endpoint
   AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=your_deployment_name
   
   # New credentials for embeddings and search
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=your_embedding_model_deployment
   AZURE_AI_SEARCH_ENDPOINT=your_search_endpoint
   AZURE_AI_SEARCH_API_KEY=your_search_api_key
   AZURE_AI_SEARCH_INDEX_NAME=your_search_index_name
   ```
   These additional credentials allow your application to access the embedding model and Azure AI Search service, which are essential for implementing vector search.

## Step 1: Create Your Enhanced RAG Application

Start by creating a new file named `sk_rag_02.py`. The easiest way is to clone your existing implementation:

```bash
cp sk_rag_01.py sk_rag_02.py
```

## Step 2: Add Imports for Vector Search and Embeddings

Update the imports section at the top of your file to include the necessary libraries for vector search:

```python
import asyncio
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
```

**What & Why**: We're adding new imports to support vector search capabilities. The `AzureOpenAI` client is used to generate embeddings (vector representations) of text. The `SearchClient` and `VectorizedQuery` classes from Azure Search enable us to perform semantic searches based on these vector representations. This approach allows us to find documents that are conceptually similar to the user query, not just those that contain matching keywords.

## Step 3: Initialize Vector Search Components

After initializing your Semantic Kernel (keeping the existing code), add code to set up the embedding client and search client:

```python
# Initialize the embeddings client
embeddings_client = AzureOpenAI(azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
                                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                                api_version="2024-10-21")

search_client = SearchClient(endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"), 
                             index_name=os.getenv("AZURE_AI_SEARCH_INDEX_NAME"),
                             credential=AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_API_KEY")))
```

**What & Why**: These clients connect to Azure services that power our vector search. The embedding client converts text to high-dimensional vectors that capture semantic meaning. The search client communicates with Azure AI Search, where your documents are stored and indexed. Together, they enable semantic search - finding documents based on meaning rather than exact keyword matches. This is significantly more powerful than traditional search methods, especially for natural language queries.

## Step 4: Update the Main Function for Vector Search

Replace the content of your `main()` function with the following code:

```python
async def main():
    # Initialize the chat completion service and execution settings
    chat_completion_service = kernel.get_service(service_id="azure-chat-completion")
    execution_settings = kernel.get_prompt_execution_settings_from_service_id("azure-chat-completion")
    system_prompt = "You are an AI assistant that provides accurate and detailed answers to questions about a vehicle's features, maintenance, and troubleshooting."
    user_question = "What is the part number for the oil filter?"

    # Step 1: Get the embeddings for the user question
    query_embeddings = embeddings_client.embeddings.create(input=user_question, model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")).data[0].embedding

    # Step 2: Perform a vector search in with Azure AI Search
    vector_query = VectorizedQuery(vector=query_embeddings, k_nearest_neighbors=5, fields="text_vector")
    search_results = search_client.search(search_text=None, vector_queries=[vector_query], select=["title", "chunk"], top=10)
    results = []
    for result in search_results:
        source = result.get("title", "Unknown Source")
        content = result.get("chunk", "No Content")
        results.append(f"<source><name>{source}</name><content>{content}</content></source>")
    results = "\n".join(results)

    # Step 3: Generate a response using the chat completion service
    print("\n=== " + user_question + "===")
    chat_history_1 = ChatHistory()
    chat_history_1.add_system_message(system_prompt)
    user_message = ("Sources:\n\n" + results + "\n\n" + "\n\nQuestion: " + user_question)
    chat_history_1.add_user_message(user_message)
    
    result_1 = await chat_completion_service.get_chat_message_content(chat_history=chat_history_1, settings=execution_settings)
    print(result_1)
```

**What & Why**: This updated function replaces our hardcoded documents with dynamically retrieved content from Azure AI Search. Here's what's happening:

1. **Embedding Generation**: We convert the user's question into a vector representation using Azure OpenAI's embedding model.
2. **Vector Search**: We use this vector to search our document index for semantically similar content, requesting the 5 most relevant matches (`k_nearest_neighbors=5`).
3. **Result Formatting**: We extract and format the search results, preserving the source document titles for attribution.
4. **RAG Integration**: Finally, we use these dynamically retrieved documents in our RAG pattern, providing them as context for the LLM to generate an accurate response.

The system prompt has been updated to focus on vehicle information, as we're assuming the search index contains vehicle documentation. This demonstrates how vector search can work with specialized knowledge domains.

## Step 5: Understanding Vector Search Parameters

The vector search operation includes several important parameters:

```python
vector_query = VectorizedQuery(vector=query_embeddings, k_nearest_neighbors=5, fields="text_vector")
search_results = search_client.search(search_text=None, vector_queries=[vector_query], select=["title", "chunk"], top=10)
```

**What & Why**: 
- `vector=query_embeddings`: The vector representation of the user's question.
- `k_nearest_neighbors=5`: Retrieves the 5 most semantically similar documents.
- `fields="text_vector"`: Specifies which vector field in the index to search against.
- `select=["title", "chunk"]`: Retrieves only these fields from the matching documents.
- `top=10`: Limits results to 10 documents maximum.

These parameters allow fine-tuning of the retrieval process. You might adjust `k_nearest_neighbors` or `top` based on your specific needs - more results provide more context but increase token usage.

## Step 6: Result Formatting for the LLM

Note how we format the search results before sending them to the LLM:

```python
results = []
for result in search_results:
    source = result.get("title", "Unknown Source")
    content = result.get("chunk", "No Content")
    results.append(f"<source><name>{source}</name><content>{content}</content></source>")
results = "\n".join(results)
```

**What & Why**: We're using XML-like formatting to clearly structure the information for the LLM. This helps the model understand what content comes from which source, enabling it to provide proper attribution in its response. Structured formatting improves the model's ability to parse and utilize the retrieved information effectively.

## Running the Enhanced Application

To run your vector search-enabled RAG application:

```bash
python sk_rag_02.py
```

You should see a response that answers the user's question about the oil filter part number, based on relevant information retrieved from your Azure AI Search index.

## Understanding Azure AI Search Index Requirements

For this lab to work, your Azure AI Search index should have:

1. A field containing vector embeddings (named `text_vector` in our example)
2. Fields for document content (`chunk`) and source attribution (`title`)
3. Vector search capabilities enabled

The index creation process is not covered in this lab, but typically involves:
- Processing source documents into chunks
- Generating embeddings for each chunk
- Creating and populating an Azure AI Search index with these chunks and their embeddings

## Challenges to Try

1. Modify the system prompt to better suit your specific document domain

## Next Steps

- Deploy your RAG application as an API

