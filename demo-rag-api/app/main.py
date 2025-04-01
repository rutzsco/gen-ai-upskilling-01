from fastapi import FastAPI
from app.routes.workflow import router as workflow_router
from app.routes.status import router as status_router
import logging
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Instrumenting the requests library for OpenTelemetry tracing
RequestsInstrumentor().instrument()

# FastAPI app setup
app = FastAPI(title="RAG API with Semantic Kernel")
app.include_router(workflow_router)
app.include_router(status_router)
FastAPIInstrumentor.instrument_app(app)

# Basic config settings
@app.get("/")
async def root():
    return {"message": "Welcome to the RAG API with Semantic Kernel"}