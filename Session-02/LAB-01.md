# Lab 01: Building a RAG Application with Semantic Kernel

## Overview
In this lab, you will build a Retrieval-Augmented Generation (RAG) application using Semantic Kernel and Azure OpenAI. The application will answer questions about corporate travel policies by referencing specific documents and providing informed responses.

## Prerequisites
- Python 3.8 or later
- Azure OpenAI account and API access
- Basic understanding of Python programming
- Visual Studio Code or your preferred code editor

## Setup

1. Install required packages:
   ```bash
   pip install semantic-kernel python-dotenv
   ```
   These packages provide the foundation for our RAG application. Semantic Kernel offers a unified interface for AI services, while python-dotenv helps manage environment variables securely.

2. Confirm you have a `.env` file with your Azure OpenAI credentials:
   ```
   AZURE_OPENAI_API_KEY=your_api_key
   AZURE_OPENAI_ENDPOINT=your_endpoint
   AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=your_deployment_name
   ```

## Step 1: Set Up the Basic Semantic Kernel Environment

Create a new file named `sk_rag_01.py` and add the following imports and configuration code:

```python
import asyncio
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory
import os
from dotenv import load_dotenv

# Initialize the Kernel with Azure OpenAI credentials
kernel = sk.Kernel()

# Load environment variables from .env file
load_dotenv()
deployment_name = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
api_key = os.environ.get("AZURE_OPENAI_API_KEY")

# Register the chat service with Semantic Kernel
chat_completion_service = AzureChatCompletion(
    deployment_name=deployment_name,  
    api_key=api_key,
    endpoint=endpoint, 
    service_id="azure-chat-completion"
)
kernel.add_service(chat_completion_service)
```

**What & Why**: This step establishes the foundation of our application. Semantic Kernel acts as the orchestration layer between your application and AI services. The `Kernel` is the central object that manages services and plugins. We're configuring Azure OpenAI as our LLM provider and registering it with the kernel so it can be used throughout the application. The `service_id` parameter allows us to reference this service later by name.

## Step 2: Create the Main Function Structure

Add an asynchronous main function structure to your code:

```python
async def main():
    # We'll add content here in the next steps
    pass

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
```

**What & Why**: We're using an asynchronous function because Semantic Kernel operations are predominantly asynchronous. This pattern allows the application to efficiently handle I/O-bound operations like API calls without blocking the main thread. The `if __name__ == "__main__"` block ensures the code only runs when the script is executed directly, not when imported as a module.

## Step 3: Set Up the Chat Service and User Query

Inside the main function, add code to initialize the chat service and define the user question:

```python
# Initialize the chat completion service and execution settings
chat_completion_service = kernel.get_service(service_id="azure-chat-completion")
execution_settings = kernel.get_prompt_execution_settings_from_service_id("azure-chat-completion")
system_prompt = "You are an AI assistant that provides clear and efficient guidance on corporate travel policies, booking, expense management, and travel logistics."
user_question = "Will I be reimbursed for a first class ticket?"
```

**What & Why**: Here we're retrieving the previously registered chat service and configuring execution settings. The `execution_settings` control parameters like temperature and max tokens for the completion. The system prompt is crucial for RAG applications as it sets the context and behavior for the AI assistant. It defines the role and expertise of the assistant, ensuring responses remain focused on the domain (travel policies). The user question represents the query we want to answer using our RAG approach.

## Step 4: Prepare Sample Documents for RAG

Add the following code to define sample documents that will serve as the knowledge base:

```python
# Prepare sample documents
docs = [
   "Travel Policy: Employees are reimbursed for airfare if they fly economy class. Flights must be booked at least 2 weeks in advance.",
   "Country Info - France: France is a popular tourist destination known for its cuisine and landmarks like the Eiffel Tower. The currency is the Euro.",
   "Country Info - Japan: Japan is known for its modern cities and traditional culture. The capital is Tokyo and the currency is the Yen."
]
```

**What & Why**: This represents our knowledge base - the "retrieval" part of RAG. In a production environment, these documents would typically come from a document store, database, or vector database. We include both relevant information (travel policy) and some peripheral information (country guides) to demonstrate how the LLM can focus on what's relevant to the question. The first document contains the key information needed to answer the user's question about first-class ticket reimbursement.

## Step 5: Implement RAG Using Chat History

Add code to create a chat history with the system prompt and construct the user message with the sample documents:

```python
print("\n=== " + user_question + "===")
chat_history_1 = ChatHistory()
chat_history_1.add_system_message(system_prompt)

# Updated user message with sample documents as sources and a relevant query
user_message = (
    "Using the following sources:\n\n" +
    "\n\n".join(docs) +
    "\n\nAnswer the question: " + user_question
)

chat_history_1.add_user_message(user_message)
```

**What & Why**: This step combines the retrieved documents with the user query - the core of the RAG pattern. The `ChatHistory` object manages the conversation context. First, we add the system message to set the assistant's role. Then we construct a special user message that includes both our retrieved documents and the user's question. This approach of including documents directly in the prompt is called "in-context RAG" - we're providing the AI with the exact information it needs to answer accurately, rather than relying on its pre-trained knowledge which might be outdated or incomplete.

## Step 6: Get the Response and Display Results

Add code to get the response from the chat service and display it:

```python
result_1 = await chat_completion_service.get_chat_message_content(
    chat_history=chat_history_1,
    settings=execution_settings,
)
print(result_1)
```

**What & Why**: In this final step, we send our prepared chat history (containing the system prompt, documents, and user question) to the LLM and await the response. The `get_chat_message_content` method handles all the communication with the Azure OpenAI service. The model processes the provided context and generates a response that's grounded in the supplied documents. This is the "generation" part of RAG - the model synthesizes an answer based on the retrieved information rather than just relying on its training data.

## Running the Application

To run your application, execute the following command in your terminal:

```bash
python sk_rag_01.py
```

You should see a response that answers the user's question about being reimbursed for a first-class ticket based on the provided travel policy document. The response should indicate that only economy class tickets are reimbursable according to the policy.

## Understanding the Code

- **Semantic Kernel**: A lightweight SDK that integrates Large Language Models (LLMs) into your applications.
- **RAG (Retrieval-Augmented Generation)**: A technique that enhances LLM responses by providing relevant documents or information from a knowledge base.
- **ChatHistory**: A Semantic Kernel component that maintains the conversation context between the user and the AI.
- **System Prompt**: Sets the behavior and role of the AI assistant.

## Challenges to Try

1. Add more travel policy documents to the knowledge base
2. Implement a function to load documents from files instead of hardcoding them
3. Create a more interactive experience by accepting user input from the console
4. Add error handling for API failures or invalid responses

## Next Steps

- Learn how to implement vector search for more efficient document retrieval
- Explore embedding models to create semantic representations of documents

