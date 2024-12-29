import ast
import logging
from time import time
from typing import Dict, Any, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic.v1 import BaseModel
from langchain_core.tools import StructuredTool

from query import get_anthropic_client, get_llm_client_info, AvailableModels
from query.context import LLMContextManager
from .custom_exception import RetryablePipelineStepException
from query.pipeline import PipelineStep, PipelineStepOutput
from query.model import PipelineStepType
from .yaml import YamlOutputParser


class NaturalLanguageToSQLGenerationPipelineStep(PipelineStep):
    """
    A pipeline step for generating SQL queries from natural language input, handling SQL execution errors,
    and interacting with database schema and tools.
    """

    class Output(BaseModel):
        reply: str
        sql: Optional[str]

    def __init__(self):
        super().__init__()
        self.name = "NaturalLanguageToSQLGenerationPipelineStep"
        self.model = AvailableModels.SONNET

        self.completion_tokens: int = None
        self.prompt_tokens: int = None

        self.start_time: float = None
        self.end_time: float = None

        self.prompt = """
        You are a skilled data analyst for a data warehouse Q&A system. Your primary responsibility involves crafting SQL queries based on user requirements and the existing database schema. Occasionally, you may also need to review and correct previously generated SQL queries and resolve SQL execution, ensuring precise and proper SQL query formulation for the specified database.

        Always assume that when users ask to search for a specific string, they expect flexible matching. Implement database-specific fuzzy search functionality without explicitly mentioning it to the user. Use the appropriate fuzzy matching function for the database being queried. If the specific extensions or functions are not available, fall back to using a combination of techniques for a more flexible match.

        Here are the preferred fuzzy search implementations for different databases (non-exhaustive list, use your judgement based on database), with advanced fallbacks:

        1. For PostgreSQL:
           Preferred (using pg_trgm extension):
           ```sql
           SELECT * FROM table 
           WHERE column % 'search_term' 
           ORDER BY similarity(column, 'search_term') DESC 
           LIMIT 10;
           ```
           Fallback:
           ```sql
           SELECT * FROM table 
           WHERE LOWER(column) LIKE LOWER('%' || REPLACE(REPLACE(REPLACE('search_term', '_', '\\_'), '%', '\\%'), ' ', '%') || '%')
           OR LOWER(column) LIKE LOWER('%' || REGEXP_REPLACE('search_term', '(.)', '%\\1') || '%')
           ORDER BY 
             CASE 
               WHEN LOWER(column) = LOWER('search_term') THEN 0
               WHEN LOWER(column) LIKE LOWER('search_term%') THEN 1
               WHEN LOWER(column) LIKE LOWER('%search_term%') THEN 2
               ELSE 3
             END,
             LENGTH(column)
           LIMIT 10;
           ```

        2. For MySQL:
           Preferred (if SOUNDEX or LEVENSHTEIN are available):
           ```sql
           SELECT * FROM table 
           WHERE SOUNDEX(column) = SOUNDEX('search_term') 
           ORDER BY LEVENSHTEIN(column, 'search_term') 
           LIMIT 10;
           ```
           Fallback:
           ```sql
           SELECT * FROM table 
           WHERE LOWER(column) LIKE LOWER(CONCAT('%', REPLACE(REPLACE(REPLACE('search_term', '_', '\\_'), '%', '\\%'), ' ', '%'), '%'))
           OR LOWER(column) REGEXP LOWER(CONCAT('^', REPLACE('search_term', ' ', '.*'), '|', REPLACE('search_term', ' ', '.*')))
           ORDER BY 
             CASE 
               WHEN LOWER(column) = LOWER('search_term') THEN 0
               WHEN LOWER(column) LIKE LOWER(CONCAT('search_term', '%')) THEN 1
               WHEN LOWER(column) LIKE LOWER(CONCAT('%', 'search_term', '%')) THEN 2
               ELSE 3
             END,
             LENGTH(column)
           LIMIT 10;
           ``` 

        3. For SQL Server:
           Preferred (using SOUNDEX or full-text search):
           ```sql
           SELECT TOP 10 * FROM table 
           WHERE SOUNDEX(column) = SOUNDEX('search_term') 
           ORDER BY DIFFERENCE(column, 'search_term') DESC;
           ```
           Fallback:
           ```sql
           SELECT TOP 10 * FROM table 
           WHERE LOWER(column) LIKE LOWER('%' + REPLACE(REPLACE(REPLACE('search_term', '_', '[_]'), '%', '[%]'), ' ', '%') + '%')
           OR LOWER(column) LIKE LOWER('%' + STUFF(REPLACE('search_term', ' ', ''), 1, 0, '%') + '%')
           ORDER BY 
             CASE 
               WHEN LOWER(column) = LOWER('search_term') THEN 0
               WHEN LOWER(column) LIKE LOWER('search_term' + '%') THEN 1
               WHEN LOWER(column) LIKE LOWER('%' + 'search_term' + '%') THEN 2
               ELSE 3
             END,
             LEN(column);
           ```

        These fallback queries provide more flexible matching by:
        1. Using case-insensitive comparison (LOWER function)
        2. Escaping special characters like '%' and '_'
        3. Allowing for partial word matches (replacing spaces with '%')
        4. Using regular expressions or LIKE patterns for more flexible matching
        5. Ordering results by relevance (exact match, starts with, contains, etc.)

        Adjust the query structure and result limit as needed based on the specific use case and desired precision.

        For all your responses, consistently use the following YAML format:

        ```yaml 
        reply: | 
         <your complete response to the user in plain text or markdown. Note you MUST USE block scalar indicator (|)>
        sql: |
            <the SQL query you crafted, structured over multiple lines for clarity>
        ```
          
        Examples:
        - User question: "What is the sales trend over time?"
            Response:
            ```yaml 
            reply: | 
             I can help you with that. Here's a query to show the sales trend over time
            sql: |
                SELECT date,
                    SUM(sales) AS total_sales
                FROM sales_data
                GROUP BY date
                ORDER BY date
            ``` 

        - User question: "Can you give me the sales by location?"
            Response:
            ```yaml 
            reply: | 
             I apologize, but it seems the sales_data schema doesn't contain geographic information. Is there another way I can help you analyze the sales data?
            sql: 
            ``` 

        - User question: "show me product 'labtob'"
            Response (assuming PostgreSQL with pg_trgm available):
            ```yaml
            reply: | 
             Here are the products matching your search for 'labtob', ordered by relevance
            sql: |
                SELECT product_name, price
                FROM products
                WHERE product_name % 'labtob'
                ORDER BY similarity(product_name, 'labtob') DESC
                LIMIT 10
            ```

            Alternative response (fallback if pg_trgm is not available):
            ```yaml
            reply: | 
             Here are the products matching your search for 'labtob', ordered by relevance
            sql: |
                SELECT product_name, price
                FROM products
                WHERE LOWER(product_name) LIKE LOWER('%' || REPLACE(REPLACE(REPLACE('labtob', '_', '\\_'), '%', '\\%'), ' ', '%') || '%')
                OR LOWER(product_name) LIKE LOWER('%' || REGEXP_REPLACE('labtob', '(.)', '%\\1') || '%')
                ORDER BY 
                  CASE 
                    WHEN LOWER(product_name) = LOWER('labtob') THEN 0
                    WHEN LOWER(product_name) LIKE LOWER('labtob%') THEN 1
                    WHEN LOWER(product_name) LIKE LOWER('%labtob%') THEN 2
                    ELSE 3
                  END,
                  LENGTH(product_name)
                LIMIT 10
            ```

        Remember to use fuzzy matching when possible, falling back to these more advanced string matching techniques when necessary. Adjust the function and parameters based on the specific database being used. Provide clear and concise responses to the user without mentioning the technical details of the search method.
        
        """

    def get_type(self) -> PipelineStepType:
        return PipelineStepType.NATURAL_LANGUAGE_TO_SQL

    def get_name(self) -> str:
        return self.name

    class SQLExecutionError(Exception):
        def __init__(self, sql, message):
            self.message = message
            self.sql = sql

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
        :param data: must contain the `schema` (database schema getter tool name) and the `requirement` (user requirement)
        :return:
        """
        try:
            # llm = get_azure_openai_client(self.model)
            # llm = get_openai_client(self.model)
            # llm = get_groq_client(self.model)
            llm = get_anthropic_client(self.model)

            database_schema = data.get("schema_tool_name")
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
                    (
                        "system",
                        "{system_prompt}\n\nDatabase Schema DDL with Instructions: {database_schema_ddl}",
                    ),
                    ("human", "Visualisation requirement: {requirement}"),
                    ("human", "Previous SQL you generated: {previous_sql}"),
                    (
                        "human",
                        "Error when executing previous SQL: {error_with_previous_sql}",
                    ),
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

            # split the schema_tool name to get the schema name
            schema_name = database_schema.split("SchemaGetter")[0]
            # filter the sql executor from the tools
            sql_executor = next(
                (tool for tool in tools if tool.name == schema_name + "QueryExecutor"),
                None,
            )
            schema = schema_tool.run({})

            # Retry to generate sql 3 times in case of sql error
            iterations = 0
            previous_sql = ""
            previous_error = ""

            self.completion_tokens = 0
            self.prompt_tokens = 0

            self.start_time = time()

            while iterations < 3:
                try:
                    return self.generate_sql(
                        requirement,
                        schema,
                        sql_executor,
                        llm,
                        prompt_template,
                        previous_sql,
                        previous_error,
                    )
                except self.SQLExecutionError as e:
                    logging.error(f"Error: {e}")
                    previous_sql = e.sql
                    previous_error = e.message
                    iterations += 1

            statistics = {
                "model_name": self.model,
                "client_info": get_llm_client_info("SONNET"),
                "completion_tokens": self.completion_tokens,
                "prompt_tokens": self.prompt_tokens,
                "call_start_time": self.start_time,
                "call_end_time": time(),
                "is_streaming_output": llm.streaming,
                "step_type": self.get_type(),
            }

            return PipelineStepOutput(
                step_type=self.get_type(),
                data={},
                input={"requirement": requirement},
                error=Exception(previous_error),
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

    def generate_sql(
        self,
        visualisation_requirement: str,
        schema: str,
        sql_executor,
        llm,
        prompt_template: ChatPromptTemplate,
        previous_sql: str = "",
        previous_error: str = "",
    ) -> PipelineStepOutput:
        result = llm.invoke(
            prompt_template.format(
                system_prompt=self.prompt,
                requirement=visualisation_requirement,
                database_schema_ddl=schema,
                previous_sql=previous_sql,
                error_with_previous_sql=previous_error,
            )
        )

        input = {"requirement": visualisation_requirement}
        raw_data = result.content
        parser = YamlOutputParser(pydantic_object=self.Output)
        try:
            output = parser.invoke(raw_data)
        except Exception as e:
            logging.error(f"Error in parsing output - {str(e)}")
            return PipelineStepOutput(
                step_type=self.get_type(),
                raw_data=raw_data,
                data={},
                input=input,
                error=RetryablePipelineStepException(
                    "Error in parsing output: " + str(e)
                ),
                terminal=True,
            )

        self.completion_tokens = (
            self.completion_tokens + result.usage_metadata["output_tokens"]
        )
        self.prompt_tokens = self.prompt_tokens + result.usage_metadata["input_tokens"]

        statistics = {
            "model_name": self.model,
            "client_info": get_llm_client_info("SONNET"),
            "completion_tokens": self.completion_tokens,
            "prompt_tokens": self.prompt_tokens,
            "call_start_time": self.start_time,
            "call_end_time": time(),
            "is_streaming_output": llm.streaming,
            "step_type": self.get_type(),
        }

        if not output.sql:
            return PipelineStepOutput(
                step_type=self.get_type(),
                raw_data=raw_data,
                data=output.dict(),
                input=input,
                error=RetryablePipelineStepException("Failed to generate sql query"),
                terminal=True,
            )

        # Verify that execute gets successfully executed
        res = sql_executor.run(output.sql)
        if self.detect_sql_error(res):
            raise self.SQLExecutionError(output.sql, res)
        output = output.dict()
        try:
            output["sql_data"] = ast.literal_eval(res)[:10]
        except:
            output["sql_data"] = None

        return PipelineStepOutput(
            step_type=self.get_type(),
            data=output,
            input=input,
            raw_data=raw_data,
            terminal=False,
            statistics=statistics,
        )

    def detect_sql_error(self, err_str: str) -> bool:
        if "Background on this error at: https://sqlalche.me" in err_str:
            return True
        return False
