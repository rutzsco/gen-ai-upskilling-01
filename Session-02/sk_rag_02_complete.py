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

# Initialize the embeddings client
embeddings_client = AzureOpenAI(azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
                                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                                api_version="2024-10-21")

search_client = SearchClient(endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"), 
                             index_name=os.getenv("AZURE_AI_SEARCH_INDEX_NAME"),
                             credential=AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_API_KEY")))
async def main():

    # Initialize the chat completion service and execution settings
    chat_completion_service = kernel.get_service(service_id="azure-chat-completion")
    execution_settings = kernel.get_prompt_execution_settings_from_service_id("azure-chat-completion")
    system_prompt = "You are an AI assistant that provides accurate and detailed answers to questions about a vehicle’s features, maintenance, and troubleshooting."
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

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
