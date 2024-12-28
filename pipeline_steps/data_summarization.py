import logging
from time import time
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from pydantic.v1 import BaseModel

from connector.service import execute_custom_query
from db import SessionLocal
from query import get_llm_client_info, AvailableModels, get_anthropic_client
from query.context import LLMContextManager
from custom_exception import RetryablePipelineStepException
from query.pipeline import PipelineStep, PipelineStepOutput
from query.model import PipelineStepType
from .util import tuple_to_csv, sql_data_to_csv
from .yaml import YamlOutputParser


class DataSummarizationPipelineStep(PipelineStep):
    """
    A pipeline step for generating data summaries from SQL queries, handling data summarization errors,
    and interacting with database schema and tools.
    """

    def __init__(self):
        super().__init__()
        self.name = "DataSummarizationPipelineStep"
        self.model = AvailableModels.SONNET

        self.prompt = """
        You are a great data analyst for a data warehouse Q&A system. Your job is to come up with the data summary based on the user's requirement, sql query data, 
        and the given schema of the database. Only part of complete data is provided to you. If sql query data is empty provide summary based on the sql query 
        and database schema in descriptive way by mentioning no data available in db for current query.

        SUMMARY SHOULD BE IN MARKDOWN FORMAT BY HIGHLIGHT IMPORTANT DETAILS.

        You MUST always answer in the following format:
        ```yaml
        reply: <your entire reply to the user goes here in escaped github flavoured markdown format>
        data_summary: |
            <summarization of sql query's data in few paragraphs in escaped github flavoured markdown format>
        ```

        hypothetical examples:
        1. Summary with SQL data (Note: Database schema has enough information to generate summary)
        SQL Query: SELECT products.product_category_name, SUM(order_payments.payment_value) AS total_payments FROM order_items INNER JOIN products ON order_items.product_id = products.product_id INNER JOIN order_payments ON order_items.order_id = order_payments.order_id GROUP BY products.product_category_name ORDER BY total_payments DESC
        Response:
        ```yaml
        reply: Here is the data summary for the given query's data.
        data_summary: |
            The SQL query retrieves the top product IDs, product category names, and their corresponding total payment values from the 'order_items', 'products', and 'order_payments' tables.
            #### Top-selling Products by Total Payments
            The data is grouped by product ID and product category name, and sorted in descending order of total payments.
            The summary shows that the top-selling products are mainly from the categories of **'telefonia_fixa'**, **'beleza_saude'**, **'ferramentas_jardim'**, and **'informatica_acessorios'**, with total payment values ranging from approximately **109,312** to **58,962**.
            This data can help identify the most profitable products and categories, enabling the company to optimize its product offerings and marketing strategies. 
        ```

        2.Summary without SQL data (Note: Database schema has enough information to generate summary)
        SQL query: SELECT p.product_id, p.product_category_name, SUM(op.payment_value) as total_payments FROM products p JOIN order_items oi ON p.product_id = oi.product_id JOIN order_payments op ON oi.order_id = op.order_id WHERE p.product_category_name ILIKE '%macbook%' OR p.product_id ILIKE '%macbook%' GROUP BY p.product_id, p.product_category_name ORDER BY total_payments DESC  LIMIT 10
        Response:
        ```yaml
        reply: Here is the data summary for the given query's data and sql query.
        data_summary: |
            The SQL query retrieves the top 10 product IDs, product category names, and their corresponding total payment values from the 'products', 'order_items', and 'order_payments' tables where the product ID or product category name contains the keyword '**macbook**'.
            #### Top MacBook Products by Total Payments
            The data is grouped by product ID and product category name, and sorted in descending order of total payments.
            **No data available in the database for the current query.**
            This data would have helped identify the top-selling MacBook products, enabling the company to optimize its product offerings and marketing strategies.
        ```
        """

    def get_type(self) -> PipelineStepType:
        return PipelineStepType.DATA_SUMMARIZATION

    def get_name(self) -> str:
        return self.name

    class Output(BaseModel):
        reply: str
        data_summary: str

    def run(
        self,
        query: str,
        expert_id: str,
        thread_id: str,
        tools: List[StructuredTool],
        context: LLMContextManager,
        data: Dict[str, Any],
    ) -> PipelineStepOutput:
        """
        :param data: must contain the `schema` (database schema getter tool name), the `requirement` (user requirement)
        :return:
        """
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

            requirement: str = data.get("requirement")
            if not requirement:
                return PipelineStepOutput(
                    step_type=self.get_type(),
                    data={},
                    input={"query": query},
                    error=Exception("No requirement found"),
                    terminal=True,
                )

            connection_id: str = data.get("connection_id")
            if not connection_id:
                return PipelineStepOutput(
                    step_type=self.get_type(),
                    data={},
                    input={},
                    error=Exception("No connection_id found"),
                    terminal=True,
                )

            sql_query = data.get("sql")

            if not sql_query:
                return PipelineStepOutput(
                    step_type=self.get_type(),
                    data={},
                    input={"query": query},
                    error=Exception("No sql found"),
                    terminal=True,
                )

            prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", "{system_prompt}"),
                    ("user", "{user_instructions}"),
                    ("user", "{user_input}"),
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

            try:
                sql_data = data.get("sql_data", None)
                if not sql_data:
                    logging.info("Got data from 'execute_custom_query'")
                    query_data = execute_custom_query(
                        connection_id, sql_query, 1, 10, None, SessionLocal()
                    )
                    formatted_sql_data = sql_data_to_csv(query_data)
                else:
                    logging.info("Got data from 'NL TO SQL' pipeline step")
                    formatted_sql_data = tuple_to_csv(sql_data)
            except Exception as e:
                return PipelineStepOutput(
                    step_type=self.get_type(),
                    data={},
                    input={"query": query, "sql_query": sql_query},
                    error=e,
                    terminal=True,
                )

            start_time = time()
            result = llm.invoke(
                prompt_template.format(
                    system_prompt=self.prompt,
                    user_instructions=context.get_user_instructions(),
                    user_input={
                        "sql_query": sql_query,
                        "sql_data": formatted_sql_data,
                        "requirement": requirement,
                    },
                    database_schema_ddl=schemas,
                )
            )
            end_time = time()

            statistics = {
                "model_name": self.model,
                "client_info": get_llm_client_info("SONNET"),
                "completion_tokens": result.response_metadata["usage"]["input_tokens"],
                "prompt_tokens": result.response_metadata["usage"]["output_tokens"],
                "call_start_time": start_time,
                "call_end_time": end_time,
                "is_streaming_output": llm.streaming,
                "step_type": self.get_type(),
            }

            input = {"query": query, "sql": sql_query, "requirement": requirement}
            raw_data = result.content
            logging.info(raw_data)
            parser = YamlOutputParser(pydantic_object=self.Output)
            try:
                output = parser.invoke(raw_data)
            except Exception:
                return PipelineStepOutput(
                    step_type=self.get_type(),
                    raw_data=raw_data,
                    data={},
                    input={"query": query},
                    error=RetryablePipelineStepException("Error in parsing output"),
                    terminal=True,
                )

            return PipelineStepOutput(
                step_type=self.get_type(),
                data=output.dict(),
                input=input,
                raw_data=raw_data,
                terminal=False,
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
