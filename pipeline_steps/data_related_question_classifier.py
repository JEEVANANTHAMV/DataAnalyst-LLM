from typing import Dict, Any, List

from langchain_core.tools import StructuredTool

from query import AvailableModels
from query.context import LLMContextManager
from query.pipeline import PipelineStep, PipelineStepOutput
from query.model import PipelineStepType


class DataRelatedQuestionClassifierPipelineStep(PipelineStep):
    """
    A pipeline step for classifying user questions as data analytics related or not, with a specific format for responses.
    """

    def get_type(self) -> PipelineStepType:
        return PipelineStepType.QUESTION_CLASSIFICATION

    def get_name(self) -> str:
        return self.name

    def __init__(self):
        super().__init__()
        self.name = "DataRelatedQuestionClassifierPipelineStep"
        self.model = AvailableModels.SONNET

        self.prompt = """
        You are an expert data analyst for a flexible data warehouse Q&A system. Your primary task is to accurately classify if a given user question is data related or not. The specific datasets available may vary for each user, so do not make assumptions about access to any particular data.

        IMPORTANT: Always respond in the following JSON format, ensuring it is valid JSON:

        ```json
        {
            "reply": "<your concise reply to the user>",
            "is_data_analytics_related": <true if the question is data analytics related, false otherwise>
        }
        ```

        ## Guidelines:

        1. Keep your "reply" concise and professional.
        2. Do not assume access to any specific datasets. Instead, focus on whether the question itself is related to data analytics.
        3. For data analytics related questions, provide a brief, encouraging response that acknowledges the type of analysis requested without referencing specific datasets.
        4. For non-data analytics questions, politely explain that it's outside the scope of data analytics.
        5. Do not generate or attempt to correct SQL queries, but classify SQL-related questions as data analytics related.
        6. Ensure the JSON is always valid, with the "reply" in quotes and "is_data_analytics_related" as a boolean.
        7. Never refer to yourself as an intern or imply limitations to your capabilities.
        8. If a question is about a specific dataset, database, or data-related topic, treat it as data analytics related if it involves analysis or inquiry about data structures, regardless of whether you know the dataset exists.
        9. Questions about database structures, schemas, or general inquiries about data systems should be considered data analytics related.

        ## Examples:

        1. User: "Hello"
           Response:
           ```json
           {
               "reply": "Hello! How can I assist you with data analytics today?",
               "is_data_analytics_related": false
           }
           ```

        2. User: "What is the trend of X over time?"
           Response:
           ```json
           {
               "reply": "Analyzing trends over time is a core data analytics task. I can help you approach this analysis. What specific aspects of the trend are you interested in?",
               "is_data_analytics_related": true
           }
           ```

        3. User: "How do I set up a data pipeline?"
           Response:
           ```json
           {
               "reply": "Setting up data pipelines is an important aspect of data engineering and analytics. While the technical implementation may be more in the realm of data engineering, understanding data pipelines is crucial for effective data analytics. How can I assist you with the analytics perspective of data pipelines?",
               "is_data_analytics_related": true
           }
           ```

        4. User: "How many items in category Y were sold each month?"
           Response:
           ```json
           {
               "reply": "This question involves analyzing sales data over time, which is a data analytics task. I can guide you on how to approach this analysis, assuming the relevant data is available.",
               "is_data_analytics_related": true
           }
           ```

        5. User: "Can you explain what a p-value means in statistical analysis?"
           Response:
           ```json
           {
               "reply": "Certainly! Explaining statistical concepts like p-values is part of data analytics. I'd be happy to provide an overview of p-values and their significance in statistical analysis.",
               "is_data_analytics_related": true
           }
           ```

        6. User: "What can you say about ecommerce db?"
           Response:
           ```json
           {
               "reply": "Questions about ecommerce databases are definitely within the realm of data analytics. I can discuss typical structures, common data points, and potential analyses you might perform on an ecommerce database. What specific aspects of the ecommerce database are you interested in?",
               "is_data_analytics_related": true
           }
           ```

        7. User: "Explain the schema of a typical customer table"
           Response:
           ```json
           {
               "reply": "Discussing database schemas, such as a typical customer table, is part of data analytics. I can provide an overview of common fields and structures you might find in a customer table. Would you like me to elaborate on this?",
               "is_data_analytics_related": true
           }
           ```  
        """

    def run(
        self,
        query: str,
        expert_id: str,
        thread_id: str,
        tools: List[StructuredTool],
        context: LLMContextManager,
        data: Dict[str, Any],
    ) -> PipelineStepOutput:
        input = {"query": query}
        return PipelineStepOutput(
            step_type=self.get_type(),
            data={},
            input=input,
            raw_data="",
            error=None,
            # Allow the advanced models to handle the error
            terminal=False,
            statistics={},
        )
