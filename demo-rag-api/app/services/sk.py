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
from app.prompt.file_service import FileService  # Import FileService from the correct module
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from app.services.retrival_plugins import RetrivalPlugin

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
    
        self.kernel = sk.Kernel()
        self.kernel.add_service(AzureChatCompletion(
            api_key=api_key,
            endpoint=endpoint,
            deployment_name=deployment_name, 
            service_id="azure-chat-completion"
        ))


        self.file_service = FileService()

        # Initialize the embeddings client
        self.embeddings_client = AzureOpenAI(azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
                                             azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                                             api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                                             api_version="2024-10-21")

        self.search_client = SearchClient(endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"), 
                                          index_name=os.getenv("AZURE_AI_SEARCH_INDEX_NAME"),
                                          credential=AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_API_KEY")))
        
        self.kernel.add_plugin(RetrivalPlugin(self.kernel), plugin_name="rag_retrival")

        pass


    async def run_rag(self, request: ChatRequest) -> str:      
        
        if not request.messages:
            raise ValueError("No messages found in request.")

        chat_completion_service = self.kernel.get_service(service_id="azure-chat-completion")
        execution_settings = self.kernel.get_prompt_execution_settings_from_service_id("azure-chat-completion")

        # Step 1: Get the search query from the user conversation
        search_system_message = self.file_service.read_file('RAGSearchSystemPrompt.txt') 
        search_chat_history = ChatHistory()
        search_chat_history.add_system_message(search_system_message)
        for message in request.messages:
            if message.role.lower() == "user":
                search_chat_history.add_user_message(message.content)
            elif message.role.lower() == "assistant":
                search_chat_history.add_assistant_message(message.content)
        search_chat_result = await chat_completion_service.get_chat_message_content(chat_history=search_chat_history, settings=execution_settings)
        search_query=f"{search_chat_result}"
        
        # Step 2: Get the embeddings for the search query
        user_question = request.messages[-1].content
        query_embeddings = self.embeddings_client.embeddings.create(input=search_query, model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")).data[0].embedding

        # Step 3: Perform a vector search in with Azure AI Search
        vector_query = VectorizedQuery(vector=query_embeddings, k_nearest_neighbors=5, fields="text_vector")
        search_results = self.search_client.search(search_text=None, vector_queries=[vector_query], select=["title", "chunk"], top=10)
        results = []
        for result in search_results:
            source = result.get("title", "Unknown Source")
            content = result.get("chunk", "No Content")
            results.append(f"<source><name>{source}</name><content>{content}</content></source>")
        results = "\n".join(results)

        # Step 4: Create a chat history and add the system message and user question
        system_message = self.file_service.read_file('RAGSystemPrompt.txt')
        chat_history_1 = ChatHistory()
        chat_history_1.add_system_message(system_message)
        for message in request.messages[:-1]:
            if message.role.lower() == "user":
                chat_history_1.add_user_message(message.content)
            elif message.role.lower() == "assistant":
                chat_history_1.add_assistant_message(message.content)


        user_message = ("Sources:\n\n" + results + "\n\n" + "\n\nQuestion: " + user_question)
        chat_history_1.add_user_message(user_message)
           
        chat_result = await chat_completion_service.get_chat_message_content(chat_history=chat_history_1, settings=execution_settings)
       
        kernel_arguments = KernelArguments()
        kernel_arguments ["diagnostics"] = []
        request_result = RequestResult(
            content=f"{chat_result}",
            execution_diagnostics=ExecutionDiagnostics(steps=kernel_arguments ["diagnostics"])
        )

        return request_result

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