# Lab 3: Calling the LLM via Semantic Kernel (SK)

Let’s walk through the updated code for **Lab 2: Calling the LLM via Semantic Kernel** step by step. This lab demonstrates how to use Semantic Kernel (SK) with Azure OpenAI to send prompts, manage chat history, and refine responses programmatically. Below, I’ll explain each part of the script and guide you through setting it up and running it.

---

## Step 1: Set Up Your Environment

Before running the code, ensure you have the following:

- Python installed (version 3.8 or higher recommended).
- Required packages installed:  
  Run the following in your terminal or command prompt:

    ```bash
    pip install semantic-kernel python-dotenv
    ```

- Azure OpenAI credentials: You’ll need your deployment name, endpoint, and API key. Store these in a `.env` file in your project folder (e.g., `sk_test` folder) like this:

    ```
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=<your-deployment-name>
    AZURE_OPENAI_ENDPOINT=<your-endpoint-url>
    AZURE_OPENAI_API_KEY=<your-api-key>
    ```

- **VS Code**: Create a new file called `sk_test.py` in your project folder.

---

## Step 2: Import Libraries and Initialize the Kernel

Here’s the first part of the code:

```python
import asyncio
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory
import os
from dotenv import load_dotenv

# Step 2: Initialize the Kernel with Azure OpenAI credentials
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

**Explanation:**

- **Imports:**
  - `asyncio`: Enables asynchronous programming, which SK uses for non-blocking calls.
  - `semantic_kernel`: The core SK library.
  - `AzureChatCompletion`: Connector for Azure OpenAI chat services.
  - `ChatHistory`: Manages conversation context (system/user/assistant messages).
  - `os` and `dotenv`: Load credentials from the `.env` file securely.

- **Kernel Initialization:**
  - `kernel = sk.Kernel()`: Creates an instance of the SK kernel.
  - `load_dotenv()`: Loads environment variables from `.env`.
  - Credentials are retrieved from the environment.

- **Chat Service Registration:**
  - `AzureChatCompletion`: Configures the Azure OpenAI connection.
  - `kernel.add_service(...)`: Registers the chat service with a unique `service_id`.

---

## Step 3: Define the Main Async Function

Define an `async def main()` function to handle prompts:

```python
# Step 3–6: Define async function to run a series of prompts
async def main():
    # Retrieve the chat completion service by id
    chat_completion_service = kernel.get_service(service_id="azure-chat-completion")
    # Retrieve the default inference settings
    execution_settings = kernel.get_prompt_execution_settings_from_service_id("azure-chat-completion")
```

**Explanation:**

- **Service Retrieval**: `kernel.get_service(...)` fetches the registered chat service.
- **Execution Settings**: Default model settings (e.g., temperature, max tokens) used by SK.

---

## Step 4: Simple Prompt with Chat History

Send a basic prompt and print the result:

```python
    print("\n=== Simple Prompt with Chat History ===")
    chat_history_1 = ChatHistory()
    chat_history_1.add_system_message("You are a helpful AI assistant specializing in programming and data science.")
    chat_history_1.add_user_message("List two applications of Python in data science.")
    
    result_1 = await chat_completion_service.get_chat_message_content(
        chat_history=chat_history_1,
        settings=execution_settings,
    )
    print(result_1)
```

**Explanation:**

- `ChatHistory()` creates a new context.
- `add_system_message`: Defines AI’s role.
- `add_user_message`: Adds a user query.
- `get_chat_message_content`: Sends and receives the AI's response.

**Sample Output:**

```
=== Simple Prompt with Chat History ===
1. Data analysis using libraries like pandas.
2. Building machine learning models with scikit-learn.
```

**Try It:** Save and run the script:

```bash
python sk_test.py
```

---

## Step 5: Multi-Turn Prompt with Chat History

Simulate a multi-turn conversation:

```python
    print("\n=== Multi-Turn Prompt with Chat History ===")
    chat_history_2 = ChatHistory()
    chat_history_2.add_system_message("You are an assistant that helps with travel advice.")
    chat_history_2.add_user_message("I want to travel to Europe. What are some budget-friendly countries to visit?")
    
    result_2 = await chat_completion_service.get_chat_message_content(
        chat_history=chat_history_2,
        settings=execution_settings,
    )
    print(result_2)
```

**Explanation:**

- Fresh `ChatHistory` with travel context.
- User asks for budget-friendly destinations.

**Sample Output:**

```
=== Multi-Turn Prompt with Chat History ===
Some budget-friendly countries to visit in Europe include Portugal, Hungary, and Poland...
```

---

## Step 6: Refine the Prompt with Formatting

Add structure to the response:

```python
    print("\n=== Iterated Prompt with Numbered List and Chat History ===")
    chat_history_3 = ChatHistory()
    chat_history_3.add_system_message("You are an assistant that helps with travel advice.")
    chat_history_3.add_user_message("I want to travel to Europe. What are some budget-friendly countries to visit? Give the answer in a numbered list.")
    
    result_3 = await chat_completion_service.get_chat_message_content(
        chat_history=chat_history_3,
        settings=execution_settings,
    )
    print(result_3)
```

**Explanation:**

- Adds prompt refinement: _“Give the answer in a numbered list”_

**Sample Output:**

```
=== Iterated Prompt with Numbered List and Chat History ===
1. Portugal - Affordable coastal towns and delicious food.
2. Hungary - Cheap thermal baths and vibrant culture in Budapest.
3. Poland - Low-cost historical sites and hearty cuisine.
```

---

## Step 7: Run the Full Script

Add the execution logic:

```python
# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
```

**Explanation:**

- Ensures `main()` only runs when the script is executed directly.
- Runs the async workflow.

---

## Full Run Output Example

```
=== Simple Prompt with Chat History ===
1. Data analysis using libraries like pandas.
2. Building machine learning models with scikit-learn.

=== Multi-Turn Prompt with Chat History ===
Some budget-friendly countries to visit in Europe include Portugal, Hungary, and Poland...

=== Iterated Prompt with Numbered List and Chat History ===
1. Portugal - Affordable coastal towns and delicious food.
2. Hungary - Cheap thermal baths and vibrant culture in Budapest.
3. Poland - Low-cost historical sites and hearty cuisine.
```

---

## Step 8: Wrap Up and Reflect

### What You Learned:

- How to set up **Semantic Kernel** with **Azure OpenAI**.
- How to use `ChatHistory` for managing message context.
- How to send and refine prompts iteratively in code.

### Next Steps:

- Experiment with more complex multi-turn conversations.
- Adjust `execution_settings` (e.g., temperature for creativity).
- Explore other SK capabilities (e.g., semantic memory, planners).

### Troubleshooting:

- **Errors?**
  - Check `.env` file credentials.
  - Ensure all packages are installed.
  - Verify your Azure OpenAI deployment is active.

- **Unexpected Output?**
  - Refine the prompt with clearer instructions or constraints.

Save your `sk_test.py` file, run it, and enjoy interacting with the AI programmatically!
