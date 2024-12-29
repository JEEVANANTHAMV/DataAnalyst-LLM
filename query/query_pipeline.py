import copy
from typing import List, Any

from langchain_core.tools import StructuredTool

from .context import LLMContextManager
from pipeline_steps.custom_exception import RetryablePipelineStepException
from .pipeline import (
    Pipeline,
    PipelineStatisticsRead,
    PipelineStep,
    PipelineResult,
    PipelineStepOutput,
    PipelineStepType,
)
from pipeline_steps.data_related_question_classifier import (
    DataRelatedQuestionClassifierPipelineStep,
)
from pipeline_steps.data_related_requirement_gathering import (
    DataRelatedRequirementGatheringPipelineStep,
)
from pipeline_steps.data_summarization import (
    DataSummarizationPipelineStep,
)
from pipeline_steps.natural_language_to_sql_generation import (
    NaturalLanguageToSQLGenerationPipelineStep,
)

"""
TODO: remove thread_id from pipeline steps and pipeline
        pipeline job is take data as input parameter, ask to AI and then return
        no connection with database, thread and expert
        All database things and thread related things will happen from expert side
"""


class QueryPipeline(Pipeline):
    """
    Steps involved in the query pipeline:
    1. DataRelatedQuestionClassifierPipelineStep
    2. DataRelatedRequirementGatheringPipelineStep
    3. NaturalLanguageToSQLGenerationPipelineStep
    4. DataSummarizationPipelineStep

    All the steps are executed in the same order and the result is returned
    """

    def get_steps(self) -> List[PipelineStep]:
        return self.pipeline_steps

    def __init__(self, pipeline_steps: List[PipelineStep]):
        super().__init__()
        self.name = "QueryPipeline"
        self.pipeline_steps = pipeline_steps

    def run(
        self,
        query: str,
        expert_id: str,
        thread_id: str,
        tools: list[StructuredTool],
        context: LLMContextManager,
        data: dict[str, Any],
    ) -> PipelineResult:
        results: List[PipelineStepOutput] = []
        statistics: List[PipelineStatisticsRead] = []
        reply = ""
        error = None
        step_type = PipelineStepType.QUESTION_CLASSIFICATION
        try:
            for step in self.pipeline_steps:
                step_type = step.get_type()
                result = step.run(query, expert_id, thread_id, tools, context, data)
                if result.statistics:
                    statistics.append(
                        PipelineStatisticsRead.model_validate(result.statistics)
                    )

                if isinstance(result.error, RetryablePipelineStepException):
                    iterator = 0
                    max_retries = result.error.retry_count
                    while iterator < max_retries:
                        iterator += 1
                        result = step.run(
                            query, expert_id, thread_id, tools, context, data
                        )
                        if result.statistics:
                            statistics.append(
                                PipelineStatisticsRead.model_validate(result.statistics)
                            )
                        if not result.error or not isinstance(
                            result.error, RetryablePipelineStepException
                        ):
                            break

                results.append(result)

                if result.data and result.data.get("reply", None):
                    reply = result.data.get("data_summary", None) or result.data.get(
                        "reply"
                    )

                if result.error:
                    return PipelineResult(
                        reply=reply,
                        steps=results,
                        is_complete=True,
                        error=result.error,
                        statistics=statistics,
                    )

                data.update(copy.deepcopy(result.data))

                if result.terminal:
                    break

            return PipelineResult(
                reply=reply,
                steps=results,
                is_complete=True,
                error=error,
                statistics=statistics,
            )
        except Exception as error:
            result = PipelineStepOutput(
                reply=reply,
                data=data,
                error=error,
                is_complete=True,
                step_type=step_type,
                input=data,
            )
            results.append(result)
            return PipelineResult(
                reply=reply,
                steps=results,
                is_complete=True,
                error=error,
                statistics=statistics,
            )

    def get_name(self) -> str:
        return self.name


query_pipeline = QueryPipeline(
    pipeline_steps=[
        DataRelatedQuestionClassifierPipelineStep(),
        DataRelatedRequirementGatheringPipelineStep(),
        NaturalLanguageToSQLGenerationPipelineStep(),
        DataSummarizationPipelineStep(),
    ]
)
