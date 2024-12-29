from typing import Optional, Callable

from pydantic import BaseModel
from sqlalchemy.orm import declarative_base
from enum import Enum
from cuid2 import cuid_wrapper
db_id_generator: Callable[[], str] = cuid_wrapper()



from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    func,
    Integer,
    Boolean,
    JSON,
    Float,
)

DBModel = declarative_base()

class QueryRequestBody(BaseModel):
    data_source_id: str
    query: Optional[str] = None
    expert_id: Optional[str] = None
    thread_id: Optional[str] = None


class ExpertPipelineThread(DBModel):
    __tablename__ = "expert_pipeline_thread"
    id = Column(String(36), primary_key=True, default=lambda: db_id_generator())
    expert_id = Column(String(36), nullable=True)
    user_id = Column(String(36), nullable=False)
    data_source_id = Column(String(36), nullable=True)
    name = Column(Text, nullable=True)
    is_archived = Column(Boolean, default=False)
    time_created = Column(DateTime, default=func.current_timestamp(), nullable=True)
    time_updated = Column(
        DateTime,
        default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
        nullable=True,
    )

    def __init__(self, id: str, name: str, expert_id: str, **kwargs):
        super().__init__(**kwargs)
        self.expert_id = expert_id
        self.name = name
        self.id = id


class ExpertPipelineMessage(DBModel):
    __tablename__ = "expert_pipeline_message"
    id = Column(String(36), primary_key=True, default=lambda: db_id_generator())
    user_id = Column(String(36), nullable=False)
    data_source_id = Column(String(36), nullable=True)
    thread_id = Column(String(36), nullable=False)
    query = Column(Text, nullable=True)
    is_archived = Column(Boolean, default=False)
    time_created = Column(DateTime, default=func.current_timestamp(), nullable=True)
    time_updated = Column(
        DateTime,
        default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
        nullable=True,
    )


class ExpertPipelineStepOutput(DBModel):
    __tablename__ = "expert_pipeline_step_output"
    id = Column(String(36), primary_key=True, default=lambda: db_id_generator())
    user_id = Column(String(36), nullable=False)
    data_source_id = Column(String(36), nullable=True)
    terminal = Column(Boolean, nullable=False)
    raw_data = Column(Text, nullable=True)
    data = Column(JSON, nullable=False)
    input = Column(JSON, nullable=False)
    error = Column(Text, nullable=True)
    step_type = Column(String(36), nullable=False)
    message_id = Column(String(36), nullable=True)
    thread_id = Column(String(36), nullable=False)
    time_created = Column(DateTime, default=func.current_timestamp(), nullable=True)
    time_updated = Column(
        DateTime,
        default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
        nullable=True,
    )

    def chat_history(self) -> str:
        paragraph = ""
        if self.step_type == 'QUESTION_CLASSIFICATION':
            if self.terminal:
                paragraph += f"{self.get_raw_data()}\n\n"
        if self.step_type == 'REQUIREMENTS_EXTRACTION':
            paragraph += f"{self.get_raw_data()}\n\n"
        if self.step_type == 'NATURAL_LANGUAGE_TO_SQL':
            paragraph += f"SQL query: {self.get_data_str('sql')}"
        if self.step_type == 'DATA_SUMMARIZATION':
            paragraph += "Here's the data summary from executing the previous SQL\n\n"
            paragraph += f"Data summary: {self.get_data_str('data_summary')}"
        return paragraph

    def get_raw_data(self):
        return self.raw_data

    def get_data_str(self, key):
        return self.data.get(key, "")


class PipelineStepStatistics(DBModel):
    __tablename__ = "pipeline_step_statistics"
    id = Column(String(36), primary_key=True, default=lambda: db_id_generator())
    step_output_id = Column(String(36), nullable=False)
    model_name = Column(String(36), nullable=False)
    client_info = Column(JSON, nullable=False)
    completion_tokens = Column(Integer, nullable=False)
    prompt_tokens = Column(Integer, nullable=False)
    call_start_time = Column(Float, nullable=False)
    call_end_time = Column(Float, nullable=False)
    is_streaming_output = Column(Boolean, nullable=True, default=False)
    step_type = Column(String(36), nullable=False)
    time_created = Column(DateTime, default=func.current_timestamp(), nullable=True)
    time_updated = Column(
        DateTime,
        default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
        nullable=True,
    )

class ExpertPipelineThreadBody(BaseModel):
    name: str

class ExpertIdEnum(Enum):
    BASE_EXPERT = "BaseExpert"
    DATA_ANALYST_EXPERT = "DataAnalystExpert"
    SALES_DATA_EXPERT = "SalesDataExpert"

class PipelineStepType(Enum):
    QUESTION_CLASSIFICATION = "question_classification"
    REQUIREMENTS_EXTRACTION = "requirements_extraction"
    NATURAL_LANGUAGE_TO_SQL = "natural_language_to_sql"
    DATA_INTERPRETATION = "data_interpretation"
    VISUALIZATION = "visualization"
    EXECUTE_SQL_QUERY = "execute_sql_query"
    EXPLAIN_SQL_QUERY = "explain_sql_query"
    DATA_SUMMARIZATION = "data_summarization"