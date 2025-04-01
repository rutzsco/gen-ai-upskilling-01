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