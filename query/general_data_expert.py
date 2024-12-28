from typing import List, Dict, Any

from fastapi import HTTPException
from langchain_core.tools import StructuredTool
from sqlalchemy.orm import Session

from model import ExpertIdEnum
from context import LLMContextManager
from pipeline import Pipeline, PipelineResult
from service import query_pipeline


class GeneralDataExpert:
    def __init__(self, pipelines: dict[str, Pipeline] = None):
        self.expert_id = ExpertIdEnum.DATA_ANALYST_EXPERT.value
        self.default_pipelines = {
            "query": query_pipeline
        }

        self.pipelines = pipelines if pipelines else self.default_pipelines

    def query(
        self,
        query: str,
        expert_id: str,
        thread_id: str,
        tools: List[StructuredTool],
        llm_context_manager: LLMContextManager,
        data: Dict[str, Any],
        session: Session,
    ) -> PipelineResult:
        try:
            if not query:
                raise HTTPException(status_code=400, detail="Query is required")

            pipeline = self.pipelines["query"]

            result: PipelineResult = pipeline.run(
                query=query,
                expert_id=expert_id,
                thread_id=thread_id,
                tools=tools,
                context=llm_context_manager,
                data={
                    "user_id": data.get("user_id", None),
                    "message_id": data.get("message_id", None),
                },
            )

            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

general_data_expert = GeneralDataExpert()
