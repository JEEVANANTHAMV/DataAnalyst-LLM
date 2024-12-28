from time import time
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from pydantic.v1 import BaseModel

from query import get_anthropic_client, get_llm_client_info, AvailableModels
from query.context import LLMContextManager
from custom_exception import RetryablePipelineStepException
from query.pipeline import PipelineStep, PipelineStepOutput
from query.model import PipelineStepType
from .yaml import YamlOutputParser


class ExplainSQLQueryPipelineStep(PipelineStep):
    """
    A pipeline step for explaining SQL queries
    """

    def __init__(self):
        super().__init__()
        self.name = "ExplainSQLQueryPipelineStep"
        self.model = AvailableModels.SONNET

        self.prompt = """
        You are a helpful SQL query explainer expert skilled in explaining SQL queries and suggest changes if sql query had any syntax errors.
        These queries can be used to generate informative charts that aid in data comprehension. Given a database schema, you can explain a SQL query.

        You MUST always answer in the following YAML format:
        ```yaml
        reply: <Super short reply to user>
        explanation: <GitHub flavored markdown based explanation.>
        ```

        Example: 01
        ```yaml
        reply: |
            I can definitely help you with that! Here is the explanation of the query.
        explanation: |
            The query selects the total sales amount for each product category from the 'sales' table.
        ```

        Example: 02
        ```yaml
        reply: |
            I cannot explain the query alone. To provide specific explanations, I would need the actual database schema details.
        explanation: |
            The query selects the total sales amount for each product category from the 'sales' table.
        ```
        """

    def get_name(self) -> str:
        return self.name

    def get_type(self) -> PipelineStepType:
        return PipelineStepType.EXPLAIN_SQL_QUERY

    class Output(BaseModel):
        reply: str
        explanation: str

    def run(
        self,
        query: str,
        expert_id: str,
        thread_id: str,
        tools: List[StructuredTool],
        context: LLMContextManager,
        data: dict,
    ) -> PipelineStepOutput:
        try:
            # llm = get_azure_openai_client(self.model)
            # llm = get_openai_client(self.model)
            # llm = get_groq_client(self.model)
            llm = get_anthropic_client(self.model)

            database_schema: str = data.get("schema_tool_name")
            if not database_schema:
                return PipelineStepOutput(
                    step_type=self.get_type(),
                    data={},
                    input={"query": query},
                    error=Exception("No schema found"),
                    terminal=True,
                )

            requirement = data.get("requirement")
            if not requirement:
                return PipelineStepOutput(
                    step_type=self.get_type(),
                    data={},
                    input={"query": query},
                    error=Exception("No requirement found"),
                    terminal=True,
                )

            prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", "{system_prompt}"),
                    ("user", "{database_schema_ddl}"),
                    ("user", "{user_instructions}"),
                    ("user", "{user_input}"),
                    ("placeholder", "{chat_history}"),
                ]
            )

            # filter tools to get the schema tool
            schema_tool = next(
                (tool for tool in tools if tool.name == database_schema), None
            )
            if not schema_tool:
                return PipelineStepOutput(
                    step_type=self.get_type(),
                    data={},
                    input={"query": query},
                    error=Exception("No schema tool found"),
                    terminal=True,
                )

            schemas = schema_tool.run({})

            chat_history = context.get_chat_history(thread_id=thread_id, depth=1)

            start_time = time()
            result = llm.invoke(
                prompt_template.format(
                    system_prompt=self.prompt,
                    chat_history=chat_history,
                    user_instructions=context.get_user_instructions(),
                    user_input=query,
                    database_schema_ddl=schemas,
                )
            )
            end_time = time()

            statistics = {
                "model_name": self.model,
                "client_info": get_llm_client_info("SONNET"),
                "completion_tokens": result.usage_metadata.get(
                    "candidates_token_count", 0
                ),
                "prompt_tokens": result.usage_metadata.get("prompt_token_count", 0),
                "call_start_time": start_time,
                "call_end_time": end_time,
                "is_streaming_output": llm.streaming,
                "step_type": self.get_type(),
            }

            input = {"query": query}

            raw_data = result.content
            parser = YamlOutputParser(pydantic_object=self.Output)
            try:
                output = parser.invoke(raw_data)
            except Exception as e:
                return PipelineStepOutput(
                    step_type=self.get_type(),
                    raw_data=raw_data,
                    data={},
                    input={"query": query},
                    error=RetryablePipelineStepException(
                        f"Error in parsing output - {str(e)}"
                    ),
                    terminal=True,
                )

            return PipelineStepOutput(
                step_type=self.get_type(),
                data=output.dict(),
                input=input,
                raw_data=raw_data,
                terminal=True,
                statistics=statistics,
            )

        except Exception as e:
            return PipelineStepOutput(
                step_type=self.get_type(),
                data={},
                input={"query": query},
                error=e,
                # Allow the advanced models to handle the error
                terminal=True,
            )
