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


# Step 3–6: Define async function to run a series of prompts
async def main():
    # Retrieve the chat completion service by id
    chat_completion_service = kernel.get_service(service_id="azure-chat-completion")
    # Retrieve the default inference settings
    execution_settings = kernel.get_prompt_execution_settings_from_service_id("azure-chat-completion")

    print("\n=== Simple Prompt with Chat History ===")
    chat_history_1 = ChatHistory()
    chat_history_1.add_system_message("You are a helpful AI assistant specializing in programming and data science.")
    chat_history_1.add_user_message("List two applications of Python in data science.")
    
    result_1 = await chat_completion_service.get_chat_message_content(
        chat_history=chat_history_1,
        settings=execution_settings,
    )
    print(result_1)

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())