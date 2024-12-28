from fastapi import APIRouter, HTTPException, Depends
import time
from sqlalchemy.orm import Session
from db import get_session
from service import Service
from general_data_expert import (
    general_data_expert,
)
from model import QueryRequestBody
from context import LLMContextManager
from pipeline import PipelineResultRead
from tools import get_db_schema_tools, get_sql_execution_tools
from service import get_connections_by_data_source
import traceback
import logging
import uuid


router = APIRouter()

@router.post("/query")
def query(
    body: QueryRequestBody,
    user,
    session: Session = Depends(get_session),
):
    start_time = time.time()  # start time
    try:
        if not body.query:
            raise HTTPException(status_code=400, detail="Query is required")

        if not body.data_source_id:
            raise HTTPException(status_code=400, detail="Data source ID is required")

        connections = get_connections_by_data_source(
            user.id, body.data_source_id, session
        )
        llm_context_manager = LLMContextManager(
            user.id, session, body.business_context_preset_id
        )
        sql_exec_tools = get_sql_execution_tools(connections)
        tools = get_db_schema_tools(
            session, connections, body.business_context_preset_id
        )
        tools.extend(sql_exec_tools)
        thread_id =  uuid.uuid4()
        message_id =  uuid.uuid4()
        Service(session).new_expert_pipeline_thread(
            id=thread_id,
            expert_id=body.expert_id or general_data_expert.expert_id,
            user_id=user.id,
            name=body.query,
            data_source_id=body.data_source_id,
        )
        Service(session).new_expert_pipeline_message(
            id=message_id,
            thread_id=thread_id,
            user_id=user.id,
            query=body.query,
            data_source_id=body.data_source_id,
        )

        data = {"user_id": user.id, "message_id": message_id}
        result = general_data_expert.query(
            body.query,
            body.expert_id or general_data_expert.expert_id,
            thread_id,
            tools,
            llm_context_manager,
            data,
            session,
        )

        Service(session).add_pipeline_result(
            user_id=user.id,
            thread_id=thread_id,
            message_id=message_id,
            pipeline_step_outputs=result.steps,
            statistics=result.statistics,
            data_source_id=body.data_source_id,
        )

        end_time = time.time()  # end time
        elapsed_time = (end_time - start_time) * 1000  # time in milliseconds
        logging.info(
            "==== Time taken for query ==== %s ==== : ====%s ms",
            body.query,
            elapsed_time,
        )

        session.commit()

        return {
            "expert_id": general_data_expert.expert_id,
            "thread_id": thread_id,
            "message_id": message_id,
            "result": PipelineResultRead(result=result),
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        error_info = traceback.format_exc()
        logging.error("Error in query: %s \n%s", e, error_info)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/{thread_id}")
def query_with_thread_id(
    body: QueryRequestBody,
    thread_id: str,
    user,
    session: Session = Depends(get_session),
):
    start_time = time.time()  # start time

    try:
        if not body.query:
            raise HTTPException(status_code=400, detail="Query is required")

        thread = Service(session).get_expert_pipeline_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        connections = get_connections_by_data_source(
            user.id, body.data_source_id, session
        )
        llm_context_manager = LLMContextManager(
            user.id, session, body.business_context_preset_id
        )
        sql_exec_tools = get_sql_execution_tools(connections)
        tools = get_db_schema_tools(
            session, connections, body.business_context_preset_id
        )
        tools.extend(sql_exec_tools)
        message_id = uuid.uuid4()
        Service(session).new_expert_pipeline_message(
            id=message_id,
            thread_id=thread_id,
            user_id=user.id,
            query=body.query,
            data_source_id=body.data_source_id,
        )

        data = {"user_id": user.id, "message_id": message_id}
        result = general_data_expert.query(
            body.query,
            body.expert_id or general_data_expert.expert_id,
            thread_id,
            tools,
            llm_context_manager,
            data,
            session,
        )

        thread.name = body.query
        Service(session).update_pipeline_thread(thread, user.id, thread_id)

        Service(session).add_pipeline_result(
            user_id=user.id,
            thread_id=thread_id,
            message_id=message_id,
            pipeline_step_outputs=result.steps,
            statistics=result.statistics,
            data_source_id=body.data_source_id,
        )

        end_time = time.time()  # end time
        elapsed_time = (end_time - start_time) * 1000  # time in milliseconds
        logging.info(
            "==== Time taken for query ==== %s ==== : ====%s ms",
            body.query,
            elapsed_time,
        )

        session.commit()

        logging.info(
            "Query endpoint: thread_id: %s, message_id: %s", thread_id, message_id
        )

        return {
            "expert_id": general_data_expert.expert_id,
            "thread_id": thread_id,
            "message_id": message_id,
            "result": PipelineResultRead(result=result),
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        error_info = traceback.format_exc()
        logging.error("Error in query: %s \n%s", e, error_info)
        raise HTTPException(status_code=500, detail=str(e))
