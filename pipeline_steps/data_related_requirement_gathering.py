from time import time
from typing import Dict, Any, List, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.callbacks.manager import get_bedrock_anthropic_callback
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from pydantic.v1 import BaseModel

from query import get_llm_client_info, AvailableModels, get_anthropic_client
from query.context import LLMContextManager
from .custom_exception import RetryablePipelineStepException
from query.pipeline import PipelineStep, PipelineStepOutput
from query.model import PipelineStepType
from query.tools import get_connection_id_from_tools
from .yaml import YamlOutputParser


class DataRelatedRequirementGatheringPipelineStep(PipelineStep):
    """
    A pipeline step for gathering data-related requirements, where the step involves
    understanding user queries, utilizing schema tools to gather necessary information, and responding
    in a structured JSON format.
    """

    def get_type(self) -> PipelineStepType:
        return PipelineStepType.REQUIREMENTS_EXTRACTION

    def get_name(self) -> str:
        return self.name

    def __init__(self):
        super().__init__()
        self.name = "DataRelatedQuestionRequirementGatheringPipelineStep"
        self.model = AvailableModels.SONNET

        self.prompt = """
        You are an elite AI Data Analyst and Query Optimization Specialist for a sophisticated data warehouse Q&A system. Your expertise lies in understanding complex data structures, resolving ambiguities, and formulating efficient query plans. You have access to database schemas via the provided database schema tools, and you excel at extracting precise data analytics requirements from conversations with non-technical users.

        A Good requirement MUST not contain any mentions of the database itself, but only the data analytics requirement. A requirement can only be considered clear if it directly translates to a SQL query in one of the accessible schemas. The requirement should include a high-level query plan in pseudo-code form, outlining the main steps to answer the query without using specific SQL syntax.

        IMPORTANT!!! Steps:
        1. Check which database schema tools are available to you. If you don't have access to any schema tools, you can't help the user. If you only have access to one database schema tool, you can assume that it is the schema the user is asking about.
        2. You MUST execute ALL relevant database schema tools to get the schemas to know what information is available. You may decide based on the chat history context here which tools to use or if that is not possible ask the user.
        3. Carefully analyze the user's query and identify potential ambiguities, including similar table names or data structures within the same schema or across different schemas.
        4. If ambiguities are present or if the query could apply to multiple tables, ALWAYS ask clarifying questions to the user (use the "reply" field of the yaml output) to understand the requirement better. Be specific about the ambiguities you've identified and provide options based on the available schemas and tables.
        5. Use the entire conversation history and table plus column context information to build context and resolve ambiguities.
        6. Reply to the user with the schema tool name and the **full** data analytics requirement with all constraints as you understand taking into account the entire conversation history if the requirement is clear. Include a high-level query plan in pseudo-code form as part of the requirement field.

        You MUST always answer in the following YAML format:
        ```
        reply: "<your entire reply to the user goes here in escaped github flavoured markdown format. Always use block and folded scalars>"
        schema_tool_name: "<the schema tool name to access the schema you are querying to answer the question. eg: if the tool name is functions.salesDataSchemaGetter then schema_tool_name=salesDataSchemaGetter. DO NOT prefix with anything extra>"
        requirement: "<Comprehensive single sentence analytical requirement from the user taking into account the conversational history, followed by a high-level query plan in pseudo-code form. Always use block and folded scalars>"
        is_requirement_clear: "<true if the requirement is clear, false if you are asking a follow-up question to understand more>"
        is_already_answered_without_further_queries: "<true ONLY if you can and have answered to satisfy user requirement based on the schema and other information available to you>"
        ```

        hypothetical examples:
            1. You have access to ONLY a single database schema tool then you can ASSUME the user is talking about that.
            user question: "What is the sales trend over time by month and region?"
            response:
            ```yaml
            reply: |
              I can help you with that as I have access to the relevant data. Let's get started.
            schema_tool_name: salesSchemaGetter
            requirement: |
              Analyze sales trend over time by month and region.
              Query Plan:
                1. Group sales data by month and region
                2. Calculate total sales for each group
                3. Order results by month (ascending) and region
                4. Return grouped and aggregated data for trend analysis
            is_requirement_clear: true
            is_already_answered_without_further_queries: false 
            ```
            2. You have access to MULTIPLE databases but based on the information/context provided by the user it is clear that they are talking about human resources.
            user question: "What is the average salary of my tech employees" (Note here that you need to first get the schema using the schema tools to know which information is actually available)
            response:
            ```yaml
            reply: |
               I can help you with that. Here's how we'll approach this query.
            schema_tool_name: humanResourcesSchemaGetter
            requirement: |
              Calculate the average salary of tech employees.
              Query Plan:
                1. Filter employees table for tech department or job titles
                2. Calculate average of salary field for filtered employees
                3. Return the calculated average salary
            is_requirement_clear: true
            is_already_answered_without_further_queries: false 
            ```
            3. User is asking a meta question about accessible schemas, and you can answer the question by executing schema tools and understanding the schemas (DO not mark requirement as clear in this case).
            user question: "What questions can be answered with our databases?"
            response:
            ```yaml
            reply: |
                ## Data Insights
                 
                According to the data available in the `ecommerce` and `sales_data` schemas, we can answer questions about:

                - **Sales trends**
                - **Customer demographics**
                - **Product performance**   
            schema_tool_name: 
            requirement: 
            is_requirement_clear: true
            is_already_answered_without_further_queries: true
            ```
            4. The requirement is not entirely clear so you need to ask follow-up questions to understand the requirement better.
            user question: "What is the sales trend overtime?" (Note here that you need to first get the schema using the schema tools to know which information is actually available)
            response:
            ```yaml
            reply: |
                ## Sales Trend Clarification
                Can you please clarify what you mean by **sales trend over time**?  
                Is it by:
                 
                - Month
                - Year
                - Region  
            schema_tool_name: salesDataSchemaGetter
            requirement: 
            is_requirement_clear: false
            is_already_answered_without_further_queries: false 
            ```
            5. You can not answer the question as the database schema does not have the required information.
            user question: "What is the location with the highest sales?" (Note here that you need to first get the schema using the schema tools to know which information is actually available)
            response:
            ```yaml
            reply: | 
                Unfortunately, I don't have the data to answer that question as the sales_data 
                schema does not store location information. Can you please provide me with a different question?
            schema_tool_name: 
            requirement: 
            is_requirement_clear: false
            is_already_answered_without_further_queries: false 
            ```
            6. Handling ambiguity between similar tables in different schemas.
            user question: "What are our top-selling products?"
            response:
            ```yaml
            reply: |
                To answer your question about top-selling products, I need to clarify which specific data you're interested in. We have two relevant tables:
                1. In the `sales` schema, there's a `product_sales` table with detailed transaction data.
                2. In the `inventory` schema, there's a `product_turnover` table with aggregate stock movement data.
                Could you please specify if you're interested in the top-selling products based on:
                a) The number of units sold
                b) The total revenue generated
                c) The frequency of restocking
                This will help me provide the most accurate information from the appropriate table.
            schema_tool_name: 
            requirement: 
            is_requirement_clear: false
            is_already_answered_without_further_queries: false 
            ```
            7. After clarification of the ambiguity in the previous example.
            user response: "I'm interested in the top-selling products based on total revenue generated."
            response:
            ```yaml
            reply: | 
              Thank you for the clarification. I'll analyze the top-selling products based on total revenue generated.
            schema_tool_name: salesSchemaGetter
            requirement: |
              Identify top-selling products based on total revenue generated.
              Query Plan:
                1. Group sales data by product ID
                2. Calculate total revenue for each product (sum of price * quantity)
                3. Order products by total revenue in descending order
                4. Limit results to top N products (e.g., top 10)
                5. Retrieve product names for the top-selling products
                6. Return list of top-selling products with their total revenue
            is_requirement_clear: true
            is_already_answered_without_further_queries: false 
            ```
            8. Handling ambiguity between similar tables within the same schema.
            user question: "What's our customer retention rate?"
            response:
            ```yaml
            reply: |
                To accurately answer your question about customer retention rate, I need to clarify which data we should use. In our `customer` schema, we have two relevant tables:
                1. `customer_transactions`: Contains individual purchase data, including customer IDs and transaction dates.
                2. `customer_subscriptions`: Tracks ongoing customer subscriptions with start and end dates.
                The retention rate calculation might differ based on which table we use:
                a) Using `customer_transactions`, we could calculate retention based on repeat purchases.
                b) Using `customer_subscriptions`, we could calculate retention based on subscription renewals.
                Which approach would you prefer for calculating the customer retention rate?
            schema_tool_name: customerSchemaGetter
            requirement: 
            is_requirement_clear: false
            is_already_answered_without_further_queries: false 
            ```
            9. After clarification of the intra-schema ambiguity.
            user response: "Let's calculate based on subscription renewals."
            response:
            ```yaml
            reply: Thank you for clarifying. I'll calculate the customer retention rate based on subscription renewals.
            schema_tool_name: customerSchemaGetter
            requirement: |
              Calculate customer retention rate based on subscription renewals.
              Query Plan:
                1. Identify distinct customers with active subscriptions in the previous period
                2. Count customers from step 1 who also have active subscriptions in the current period
                3. Divide the count from step 2 by the total count from step 1
                4. Multiply by 100 to get the retention rate percentage
                5. Return the calculated retention rate
            is_requirement_clear: true
            is_already_answered_without_further_queries: false 
            ``` 

        You MUST in all cases return the final output in the given correctly formatted valid YAML template (fill in the _blank_s). DO NOT include any additional text in the reply. User does not see those.

        ```yaml
        reply: _blank_
        schema_tool_name: _blank_
        requirement: _blank_
        is_requirement_clear: _blank_
        is_already_answered_without_further_queries: _blank_ 
        ```
         
        Please pay attention to the instructions and construct a correct YAML response always. 
         
        Do not use multi_tool_use.parallel function. If you need to call multiple functions, you should call them one at a time.
        """

    class Output(BaseModel):
        reply: str
        schema_tool_name: Optional[str]
        requirement: Optional[str]
        is_requirement_clear: bool
        is_already_answered_without_further_queries: bool

    def run(
        self,
        query: str,
        expert_id: str,
        thread_id: str,
        tools: List[StructuredTool],
        context: LLMContextManager,
        data: Dict[str, Any],
    ) -> PipelineStepOutput:
        try:
            llm = get_anthropic_client(self.model)

            chat_history = context.get_chat_history(thread_id=thread_id, depth=-1)
            business_info = None
            messages = [
                ("system", "{system_prompt}"),
                ("user", "{user_instructions}"),
                ("user", "{user_defined_business_info}") if business_info else None,
                ("user", "{user_input}"),
                ("placeholder", "{agent_scratchpad}"),
                ("placeholder", "{chat_history}"),
            ]

            prompt_template = ChatPromptTemplate.from_messages(
                [msg for msg in messages if msg is not None]
            )

            object_to_pass = {
                "system_prompt": self.prompt,
                "user_instructions": context.get_user_instructions(),
                "user_defined_business_info": business_info if business_info else None,
                "user_input": query,
                "chat_history": chat_history,
            }

            # Remove None values from object_to_pass if not needed
            object_to_pass = {k: v for k, v in object_to_pass.items() if v is not None}

            # keep only schema getter tools from tools
            tools = [tool for tool in tools if "SchemaGetter" in tool.name]
            print(llm,tools,prompt_template)
            agent = create_tool_calling_agent(llm, tools=tools, prompt=prompt_template)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                stream_runnable=False,
                handle_parsing_errors=True,
                max_iterations=5,
            )

            with get_bedrock_anthropic_callback() as cb:
                start_time = time()
                business_info = None

                agent_response = agent_executor.invoke(object_to_pass)

                end_time = time()

                statistics = {
                    "model_name": self.model,
                    "client_info": get_llm_client_info("SONNET"),
                    "completion_tokens": cb.completion_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "call_start_time": start_time,
                    "call_end_time": end_time,
                    "is_streaming_output": llm.streaming,
                    "step_type": self.get_type(),
                }

            raw_data = agent_response["output"]

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
            input = {"query": query}

            if (
                output.is_requirement_clear
                and not output.is_already_answered_without_further_queries
            ):
                connection_id = get_connection_id_from_tools(
                    output.schema_tool_name, tools
                )
                if not connection_id:
                    return PipelineStepOutput(
                        step_type=self.get_type(),
                        data=output.dict(),
                        input=input,
                        raw_data=raw_data,
                        error=RetryablePipelineStepException(
                            "AI could not decide on the database, so no connection id found"
                        ),
                        terminal=True,
                        statistics=statistics,
                    )
                out = output.dict()
                out["connection_id"] = connection_id
                return PipelineStepOutput(
                    step_type=self.get_type(),
                    data=out,
                    input=input,
                    raw_data=raw_data,
                    terminal=False,
                    statistics=statistics,
                )
            else:
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
                terminal=True,
            )
