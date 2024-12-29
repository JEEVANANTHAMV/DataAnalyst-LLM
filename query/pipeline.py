from abc import abstractmethod, ABC
from typing import Dict, Any, List

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from .context import LLMContextManager
from .model import PipelineStepType


class PipelineStatisticsRead(BaseModel):
    model_name: str
    client_info: Dict
    completion_tokens: int
    prompt_tokens: int
    call_start_time: float
    call_end_time: float
    is_streaming_output: bool
    step_type: PipelineStepType

    class Config:
        protected_namespaces = ()


class PipelineStepOutput:
    def __init__(
        self,
        step_type: PipelineStepType,
        data: Dict[str, Any],
        input: Dict[str, Any],
        raw_data: str = "",
        terminal: bool = True,
        error: Exception = None,
        **kwargs
    ):
        self.terminal = terminal
        self.raw_data = raw_data
        self.data = data
        self.input = input
        self.error = error
        self.step_type = step_type
        self.statistics = kwargs.get("statistics", None)


# abstract class for a step in the pipeline
class PipelineStep(ABC):
    @abstractmethod
    def run(
        self,
        query: str,
        expert_id: str,
        thread_id: str,
        tools: List[StructuredTool],
        context: LLMContextManager,
        data: Dict[str, Any],
    ) -> PipelineStepOutput:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_type(self) -> PipelineStepType:
        pass


class PipelineResult:
    def __init__(
        self,
        reply: str,
        steps: List[PipelineStepOutput],
        statistics: List[PipelineStatisticsRead],
        is_complete: bool = True,
        error: Exception = None,
    ):
        self.reply = reply
        self.steps = steps
        self.is_complete = is_complete
        self.error = error
        self.statistics = statistics


class Pipeline(ABC):
    @abstractmethod
    def run(
        self,
        query: str,
        expert_id: str,
        thread_id: str,
        tools: List[StructuredTool],
        context: LLMContextManager,
        data: Dict[str, Any],
    ) -> PipelineResult:
        pass

    @abstractmethod
    def get_steps(self) -> List[PipelineStep]:
        pass


class PipelineStepOutputRead(BaseModel):
    terminal: bool
    raw_data: str
    data: Dict[str, Any]
    input: Dict[str, Any]
    error: str
    step_type: str

    def __init__(self, data: PipelineStepOutput):
        super().__init__(
            terminal=data.terminal,
            data=data.data,
            raw_data=data.raw_data,
            input=data.input,
            error=str(data.error),
            step_type=data.step_type,
        )


class PipelineResultRead(BaseModel):
    reply: str
    steps: List[PipelineStepOutputRead]
    is_complete: bool
    error: str

    def __init__(self, result: PipelineResult):
        super().__init__(
            reply=result.reply,
            steps=[PipelineStepOutputRead(step) for step in result.steps],
            is_complete=result.is_complete,
            error=str(result.error),
        )
