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

async def main():

    # Initialize the chat completion service and execution settings
    chat_completion_service = kernel.get_service(service_id="azure-chat-completion")
    execution_settings = kernel.get_prompt_execution_settings_from_service_id("azure-chat-completion")
    system_prompt = "You are an AI assistant that provides clear and efficient guidance on corporate travel policies, booking, expense management, and travel logistics."
    user_question = "Will I be reimbursed for a first class ticket?"

    # Step 1: Prepare sample documents
    docs = [
       "Travel Policy: Employees are reimbursed for airfare if they fly economy class. Flights must be booked at least 2 weeks in advance.",
       "Country Info - France: France is a popular tourist destination known for its cuisine and landmarks like the Eiffel Tower. The currency is the Euro.",
       "Country Info - Japan: Japan is known for its modern cities and traditional culture. The capital is Tokyo and the currency is the Yen."
    ]

    # Step 3: Basic prompt with chat history
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
    
    result_1 = await chat_completion_service.get_chat_message_content(
        chat_history=chat_history_1,
        settings=execution_settings,
    )
    print(result_1)


# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
