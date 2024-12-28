import math
from typing import Any, List, Dict

from fastapi import HTTPException
from sqlalchemy import desc, asc
from sqlalchemy.orm import Session

from db import commit_session
from query.model import ExpertPipelineThreadBody
from query.pipeline import PipelineStepOutput, PipelineStatisticsRead
from query.model import (
    ExpertPipelineThread,
    ExpertPipelineStepOutput,
    ExpertPipelineMessage,
    PipelineStepStatistics,
)
import uuid
from connector.connector import DatabaseConnectionRead, SupportDwhDialect
from connector.service import DataSource, create_database_metadata



class Service:
    def __init__(self, session: Session):
        self.session = session

    def new_expert_pipeline_thread(
        self,
        id: str,
        expert_id: str,
        user_id: str,
        data_source_id: str,
        name: str = None,
    ) -> ExpertPipelineThread:
        thread = ExpertPipelineThread(
            id=id,
            name=name,
            expert_id=expert_id,
            user_id=user_id,
            data_source_id=data_source_id,
        )
        self.session.add(thread)
        self.session.flush()

        return thread

    def new_expert_pipeline_message(
        self, id: str, thread_id: str, user_id: str, query: str, data_source_id: str
    ) -> ExpertPipelineMessage:
        message = ExpertPipelineMessage(
            id=id,
            thread_id=thread_id,
            user_id=user_id,
            query=query,
            data_source_id=data_source_id,
        )
        self.session.add(message)
        self.session.flush()
        return message

    def get_expert_pipeline_thread(self, id: str) -> ExpertPipelineThread:
        return (
            self.session.query(ExpertPipelineThread)
            .filter_by(id=id, is_archived=False)
            .first()
        )

    def filter_expert_pipeline_output(
        self, thread_id: str, pipeline_step_type: str, message_id: str
    ) -> ExpertPipelineStepOutput:
        latest_output = (
            self.session.query(ExpertPipelineStepOutput)
            .filter_by(
                thread_id=thread_id, step_type=pipeline_step_type, message_id=message_id
            )
            .order_by(desc(ExpertPipelineStepOutput.time_created))
            .first()
        )
        return latest_output

    def get_latest_query_from_pipeline_output(self, thread_id: str) -> str:
        query_pipeline_out = (
            self.session.query(ExpertPipelineStepOutput)
            .filter_by(
                thread_id=thread_id,
                step_type='NATURAL_LANGUAGE_TO_SQL',
            )
            .first()
        )

        if (
            not query_pipeline_out
            or not query_pipeline_out.data
            or not query_pipeline_out.data.get("sql")
        ):
            raise HTTPException(
                status_code=500, detail="Expert pipeline output not  found"
            )

        return query_pipeline_out.data.get("sql")

    def add_pipeline_step_output(
        self,
        user_id: str,
        thread_id: str,
        message_id: str,
        pipeline_output: ExpertPipelineStepOutput,
    ):
        step_type = (
            pipeline_output.step_type
            if isinstance(pipeline_output.step_type, str)
            else pipeline_output.step_type.value
        )
        error = (
            pipeline_output.error
            if isinstance(pipeline_output.error, str)
            else str(pipeline_output.error)
        )
        id = uuid.uuid4()

        self.session.add(
            ExpertPipelineStepOutput(
                id=id,
                terminal=pipeline_output.terminal,
                data=pipeline_output.data,
                input=pipeline_output.input,
                error=error,
                step_type=step_type,
                message_id=message_id,
                user_id=user_id,
                thread_id=thread_id,
                raw_data=pipeline_output.raw_data,
            )
        )

        return id

    def add_pipeline_step_statistics(self, step_output_id, step_type, statistics):
        step_type = step_type if isinstance(step_type, str) else step_type.value
        for item in statistics:
            self.session.add(
                PipelineStepStatistics(
                    step_output_id=step_output_id,
                    model_name=item.get("model_name", None),
                    client_info=item.get("client_info", None),
                    completion_tokens=item.get("completion_tokens", None),
                    prompt_tokens=item.get("prompt_tokens", None),
                    call_start_time=item.get("call_start_time", None),
                    call_end_time=item.get("call_end_time", None),
                    is_streaming_output=item.get("is_streaming_output", None),
                    step_type=step_type,
                )
            )

    def add_pipeline_result(
        self,
        user_id: str,
        data_source_id: str,
        thread_id: str,
        message_id: str,
        pipeline_step_outputs: List[PipelineStepOutput],
        statistics: List[PipelineStatisticsRead],
    ):
        for item in pipeline_step_outputs:
            id = uuid.uuid4()
            self.session.add(
                ExpertPipelineStepOutput(
                    id=id,
                    terminal=item.terminal,
                    data=item.data,
                    data_source_id=data_source_id,
                    input=item.input,
                    error=(
                        item.error if isinstance(item.error, str) else str(item.error)
                    ),
                    step_type=item.step_type.value,
                    message_id=message_id,
                    user_id=user_id,
                    thread_id=thread_id,
                    raw_data=item.raw_data,
                )
            )

            for stat in statistics:
                if stat.step_type.value == item.step_type.value:
                    self.session.add(
                        PipelineStepStatistics(
                            step_output_id=id,
                            model_name=stat.model_name,
                            client_info=stat.client_info,
                            completion_tokens=stat.completion_tokens,
                            prompt_tokens=stat.prompt_tokens,
                            call_start_time=stat.call_start_time,
                            call_end_time=stat.call_end_time,
                            is_streaming_output=stat.is_streaming_output,
                            step_type=stat.step_type.value,
                        )
                    )

        commit_session(self.session)

    def get_all_pipeline_threads(
        self, data_source_id: str, user_id: str, page: int = 1, page_size=10
    ) -> List[ExpertPipelineThread]:
        offset = (page - 1) * page_size
        threads = (
            self.session.query(ExpertPipelineThread)
            .filter_by(
                user_id=user_id, is_archived=False, data_source_id=data_source_id
            )
            .order_by(desc(ExpertPipelineThread.time_created))
            .offset(offset)
            .limit(page_size)
            .all()
        )
        count = (
            self.session.query(ExpertPipelineThread)
            .filter_by(
                user_id=user_id, is_archived=False, data_source_id=data_source_id
            )
            .count()
        )

        return {
            "threads": threads,
            "count": count,
            "total_pages": math.ceil(count / page_size),
        }

    def update_pipeline_thread(
        self, data: ExpertPipelineThreadBody, user_id: str, thread_id: str
    ):
        thread = (
            self.session.query(ExpertPipelineThread)
            .filter_by(id=thread_id, user_id=user_id, is_archived=False)
            .first()
        )
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        thread.name = data.name
        commit_session(self.session)
        return thread

    def get_pipeline_thread_output(
        self, user_id: str, thread_id: str, page: int = 1, page_size=10
    ) -> List[ExpertPipelineStepOutput]:
        offset = (page - 1) * page_size
        return (
            self.session.query(ExpertPipelineStepOutput)
            .filter_by(user_id=user_id, thread_id=thread_id)
            .order_by(desc(ExpertPipelineStepOutput.time_created))
            .offset(offset)
            .limit(page_size)
            .all()
        )

    def get_user_expert_pipeline_thread(
        self, thread_id: str, user_id: str
    ) -> ExpertPipelineThread:
        """
        :param thread_id: ID of the thread
        :param user_id: ID of the user
        :return: ExpertPipelineThread

        Get a user's expert pipeline thread by ID.
        """
        return (
            self.session.query(ExpertPipelineThread)
            .filter_by(id=thread_id, user_id=user_id, is_archived=False)
            .first()
        )

    def get_messages_by_thread(
        self, thread_id: str, user_id: str, page: int = 1, page_size=10
    ):
        """
        :param thread_id: ID of the thread
        :param user_id: ID of the user
        :param page: Page number
        :param page_size: Number of messages per page
        :return:

        Get active messages of a thread.
        """
        offset = (page - 1) * page_size
        messages = (
            self.session.query(ExpertPipelineMessage)
            .filter_by(user_id=user_id, thread_id=thread_id, is_archived=False)
            .order_by(asc(ExpertPipelineMessage.time_created))
            .offset(offset)
            .limit(page_size)
            .all()
        )
        count = (
            self.session.query(ExpertPipelineMessage)
            .filter_by(user_id=user_id, thread_id=thread_id, is_archived=False)
            .count()
        )

        return messages, count, math.ceil(count / page_size)

    def get_reply(self, step_results: Dict[str, Any]):
        """
        :param step_results: list of step results
        :return: reply

        To set data_summary as message reply.
        If message has data_summary (consists pipeline step type 'data_summarization'), then return that.
        Otherwise, return the last message reply.
        """
        result = step_results[-1]

        for step_result in step_results:
            if (
                step_result.get("step_type")
                == 'DATA_SUMMARIZATION'
            ):
                if step_result.get("data") and step_result.get("data").get(
                    "data_summary"
                ):
                    return step_result["data"]["data_summary"]

        if not result.get("data") or not result.get("data").get("reply"):
            return None

        return result["data"]["reply"]

    def get_pipeline_thread_messages(
        self, user_id: str, thread_id: str, page: int = 1, page_size=10
    ) -> List[Dict]:
        pipeline_thread = self.get_user_expert_pipeline_thread(
            thread_id=thread_id, user_id=user_id
        )
        if not pipeline_thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        messages, count, total_pages = self.get_messages_by_thread(
            thread_id=thread_id, user_id=user_id, page=page, page_size=page_size
        )
        if len(messages) == 0:
            return {"count": count, "total_pages": total_pages, "messages": messages}

        formatted_messages = []
        for message in messages:
            outputs = self.get_step_outputs_for_message(message.id)

            if len(outputs) == 0:
                formatted_message = {
                    "id": message.id,
                    "reply": None,
                    "query": message.query,
                    "steps": [],
                    "time_created": message.time_created.isoformat(),
                    "time_updated": message.time_updated.isoformat(),
                }
            else:
                formatted_message = {
                    "id": message.id,
                    "reply": self.get_reply(outputs),
                    "query": message.query,
                    "steps": outputs,
                    "time_created": message.time_created.isoformat(),
                    "time_updated": message.time_updated.isoformat(),
                }
            formatted_messages.append(formatted_message)

        return {
            "thread_id": pipeline_thread.id,
            "expert_id": pipeline_thread.expert_id,
            "messages": formatted_messages,
            "count": count,
            "total_pages": total_pages,
        }

    def get_step_outputs_for_message(
        self,
        message_id: str,
    ) -> List[Dict]:
        step_outputs = (
            self.session.query(ExpertPipelineStepOutput)
            .filter_by(message_id=message_id)
            .all()
        )
        formatted_step_outputs = []
        for output in step_outputs:
            formatted_output = {
                "terminal": output.terminal,
                "data": output.data,
                "input": output.input,
                "error": output.error,
                "step_type": output.step_type,
                "time_created": output.time_created.isoformat(),
                "time_updated": output.time_updated.isoformat(),
            }
            formatted_step_outputs.append(formatted_output)

        return formatted_step_outputs

    def get_pipeline_thread_message(
        self, user_id: str, thread_id: str, message_id: str
    ) -> ExpertPipelineMessage:
        message = (
            self.session.query(ExpertPipelineMessage)
            .filter_by(user_id=user_id, thread_id=thread_id, id=message_id)
            .first()
        )
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")

        return message

    def archive_pipeline_thread_message(
        self, user_id: str, thread_id: str, message_id: str
    ):
        message = (
            self.session.query(ExpertPipelineMessage)
            .filter_by(user_id=user_id, thread_id=thread_id, id=message_id)
            .first()
        )
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")

        # archive the message
        message.is_archived = True
        commit_session(self.session)

    def archive_pipeline_thread(
        self,
        user_id: str,
        thread_id: str,
    ):
        thread = (
            self.session.query(ExpertPipelineThread)
            .filter_by(id=thread_id, user_id=user_id, is_archived=False)
            .first()
        )
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        # archive the thread
        thread.is_archived = True
        commit_session(self.session)

    def is_pipeline_thread_archived(self, user_id: str, thread_id: str):
        thread = (
            self.session.query(ExpertPipelineThread)
            .filter_by(id=thread_id, user_id=user_id)
            .first()
        )
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        return thread.is_archived

    def is_pipeline_thread_message_archived(
        self, user_id: str, thread_id: str, message_id: str
    ):
        message = (
            self.session.query(ExpertPipelineMessage)
            .filter_by(user_id=user_id, thread_id=thread_id, id=message_id)
            .first()
        )
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")

        return message.is_archived

def get_connections_by_data_source(data_source_id: str, session: Session):
    data_source = session.query(DataSource).filter_by(id=data_source_id).first()
    if not data_source:
        raise HTTPException(status_code=404, detail="Data source not found")
    connections = []
    connection = DatabaseConnectionRead(
        connection_id=uuid.uuid4(),
        db_name=data_source.database_name,
        connection_name=data_source.name,
        sql_dialect= 'postgresql',
        schema_ddl=None,
        instructions="General",
        is_owner=True,
        is_team_visible=True,
    )
    connections.append(connection)
    create_database_metadata(session, data_source)

    return connections