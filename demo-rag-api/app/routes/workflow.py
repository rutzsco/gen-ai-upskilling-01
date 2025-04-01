from fastapi import APIRouter
from pydantic import BaseModel
from app.models.api_models import ChatRequest
from app.services.sk import SemanticKernelService
import asyncio

router = APIRouter()

sk_service = SemanticKernelService()


@router.post("/rag")
async def run_rag_workflow(input_data: ChatRequest):
    """
    POST endpoint for executing a rag workflow.
    """
    result = await sk_service.run_rag(input_data)
    return {"result": result}

@router.post("/rag-agent")
async def run_rag_agent(input_data: ChatRequest):
    """
    POST endpoint for executing a rag agent.
    """
    result = await sk_service.run_rag_agent(input_data)
    return {"result": result}