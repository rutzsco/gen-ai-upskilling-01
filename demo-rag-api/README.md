# demo-ai-flows-pyhton

## Docker Build and Run

To build the Docker image:
```bash
docker build -t demo-ai-flows .
```

To run the container:
```bash
docker run -dp 8000:8000 demo-ai-flows
```

## Environment Variables

### **Azure OpenAI Configuration**
| Variable | Description | Example |
|----------|------------|---------|
| `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME` | The deployment name of the GPT-4o model on Azure OpenAI. | `gpt-4o` |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` | The deployment name for the embedding model in Azure OpenAI. Leave blank if not used. | `text-embedding-ada-002` |
| `AZURE_OPENAI_ENDPOINT` | The base URL of the Azure OpenAI service. | `https://your-openai-instance.openai.azure.com/` |
| `AZURE_OPENAI_API_KEY` | API key for authenticating requests to Azure OpenAI. | `your-api-key` |

### **Azure AI Search Configuration**
| Variable | Description | Example |
|----------|------------|---------|
| `AZURE_AI_SEARCH_API_KEY` | API key for Azure AI Search. Required if using semantic search. | `your-search-api-key` |
| `AZURE_AI_SEARCH_ENDPOINT` | Endpoint for Azure AI Search. | `https://your-search-instance.search.windows.net/` |

### **Observability & Monitoring**
| Variable | Description | Example |
|----------|------------|---------|
| `SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS` | Enables OpenTelemetry diagnostics for Semantic Kernel. | `true` |
| `SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS_SENSITIVE` | Enables sensitive telemetry logging for Semantic Kernel. Only enable this in secure environments. | `true` |

### **Application Insights (OPTIONAL)**
| Variable | Description | Example |
|----------|------------|---------|
| `APPLICATIONINSIGHTS_CONNECTION_STRING` | Connection string for Azure Application Insights. Used for monitoring application performance. | `InstrumentationKey=your-key;IngestionEndpoint=https://your-ingestion-endpoint/` |


**Sample .env**

```bash
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="gpt-4o"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME="text-embedding-3-large"
AZURE_OPENAI_ENDPOINT="https://VALUE.openai.azure.com/"
AZURE_OPENAI_API_KEY="VALUE"

AZURE_AI_SEARCH_API_KEY=""
AZURE_AI_SEARCH_ENDPOINT=""
VectorSearchIndexName_KnowledgeArticles=""

```