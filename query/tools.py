import logging
from collections import defaultdict
from typing import List, Dict, Any, Optional

import pandas as pd
from langchain.tools import tool
from langchain_core.tools import StructuredTool
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session
from re import sub, search

from connector.connector import DatabaseConnectionRead
from connector.service import (
    TableMetadata,
    query_database_connection,
    connector_from_db_name,
)

def create_list_tables_tool(engine):
    inspector = inspect(engine)

    @tool(parse_docstring=True)
    def list_tables() -> List[str]:
        """
        Lists all tables in the database.

        Returns:
            List[str]: A list of table names.
        """
        return inspector.get_table_names()

    return list_tables


def create_describe_table_tool(engine):
    inspector = inspect(engine)

    @tool(parse_docstring=True)
    def describe_table(table_name: str) -> List[Dict[str, str]] | str:
        """
        Describe a table by listing its columns and types.

        Args:
            table_name: The name of the table to describe.

        Returns:
            List[Dict[str, str]] | str: A list of dictionaries with column names and types, or an error message.
        """
        try:
            return [
                {"name": column["name"], "type": str(column["type"])}
                for column in inspector.get_columns(table_name)
            ]
        except Exception as e:
            return f"Error retrieving table information: {str(e)}"

    return describe_table


def create_column_info_tool(engine):
    inspector = inspect(engine)

    @tool(parse_docstring=True)
    def column_info(table_name: str, column_name: str) -> Dict[str, Any] | str:
        """
        Get detailed information about a specific column.

        Args:
            table_name: The name of the table.
            column_name: The name of the column.

        Returns:
            Dict[str, Any] | str: A dictionary containing column information or an error message if retrieval fails.
        """
        try:
            columns = inspector.get_columns(table_name)
            for column in columns:
                if column["name"] == column_name:
                    return {
                        "name": column["name"],
                        "type": str(column["type"]),
                        "nullable": column["nullable"],
                        "default": column["default"],
                        "primary_key": column.get("primary_key", False),
                    }
        except Exception as e:
            return f"Error retrieving column information: {str(e)}"

    return column_info


def create_sample_data_tool(engine):

    @tool(parse_docstring=True)
    def sample_data(table_name: str, limit: int = 1) -> List[Dict[str, Any]] | str:
        """
        Retrieve sample data from a table

        Args:
            table_name: The name of the table to retrieve data from.
            limit: The number of rows to retrieve.

        Returns:
            List[Dict[str, Any]] | str: A list of dictionaries containing the sample data, or an error message if the query fails.
        """

        query = f"SELECT TOP {limit} * FROM {table_name}"
        try:
            df = pd.read_sql(text(query), engine)
            return df.to_dict(orient="records")
        except Exception as e:
            return f"Error executing query: {str(e)}"

    return sample_data


def is_read_only_query(query: str) -> bool:
    write_commands = {
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "CREATE",
        "ALTER",
        "TRUNCATE",
        "REPLACE",
        "MERGE",
    }
    first_word = query.strip().split(maxsplit=1)[0].upper()
    return first_word not in write_commands


def create_execute_query_tool(engine):

    @tool(parse_docstring=True)
    def execute_sql(sql: str | None = None) -> List[Dict[str, Any]] | str:
        """
        Execute a SQL query.

        Args:
            sql: Must be a valid SQL query.

        Returns:
            List[Dict[str, Any]] | str: The result of the query.
        """
        if sql is None:
            return "Error: SQL was not provided for the tool call. Make sure to output required tool calls and parameters in the correct json format within a single line."

        try:
            df = pd.read_sql(text(sql), engine)
            # Check if more than 5 rows were returned
            if len(df) > 5:
                return "Error in executing query: returned more than 5 rows. Please add a row limit to your query."

            # Limit the result to a maximum of 5 rows
            df = df.head(5)

            return df.to_dict(orient="records")
        except Exception as e:
            return f"Error executing query: {str(e)}"

    return execute_sql


def create_fuzzy_search_in_table_tool(engine):
    inspector = inspect(engine)

    @tool(parse_docstring=True)
    def fuzzy_search_in_table(
        table_name: str, search_string: str, limit: int = 5
    ) -> List[Dict[str, Any]] | str:
        """
        Perform a fuzzy search in a table.

        Args:
            table_name: The name of the table to search in.
            search_string: The string to search for.
            limit: The maximum number of results to return. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the search results.
        """

        try:
            columns = [col["name"] for col in inspector.get_columns(table_name)]
            conditions = [
                f"DIFFERENCE(CAST({column} AS NVARCHAR(MAX)), '{search_string}') >= 3"
                for column in columns
            ]
            where_clause = " OR ".join(conditions)

            query = f"""
            SELECT TOP {limit} *,
                   (SELECT MAX(diff_score) FROM (VALUES 
                       {','.join([f"(DIFFERENCE(CAST({col} AS NVARCHAR(MAX)), '{search_string}'))" for col in columns])}
                   ) AS diff(diff_score)) AS max_diff_score
            FROM {table_name}
            WHERE {where_clause}
            ORDER BY max_diff_score DESC
            """

            results = pd.read_sql(text(query), engine)

            formatted_results = []
            for row in results:
                max_score = row.pop("max_diff_score")
                for column, value in row.items():
                    if value is not None:
                        formatted_results.append(
                            {
                                "column": column,
                                "value": str(value),
                                "score": max_score * 25,  # Scale to 0-100
                            }
                        )

            formatted_results.sort(key=lambda x: x["score"], reverse=True)
            return formatted_results[:limit]
        except Exception as e:
            return f"Error executing query: {str(e)}"

    return fuzzy_search_in_table


def merge_comparisons(comparisons):
    """
    Merge a list of comparison dictionaries into a single dictionary.
    """
    merged_result = defaultdict(list)

    for comp in comparisons:
        for key, values in comp.items():
            merged_result[key].extend(values)

    return dict(merged_result)


def compare_json_objects(obj1, obj2):
    result = {"Values Mismatch": [], "Missing Columns": []}
    for key in obj1:
        if key in obj2:
            if obj1[key] != obj2[key]:
                # print(
                #     f"Field: {key}, \nValue in Obj1: {obj1[key]}, Value in Obj2: {obj2[key]}, \nStatus: Match"
                # )
                # else:
                print(
                    f"Field: {key}, \nValue in Obj1: {obj1[key]}, Value in Obj2: {obj2[key]}, \nStatus: Mismatch"
                )
                result["Values Mismatch"].append([obj1[key], obj2[key]])
        else:
            print(f"Field: {key} is missing in the second object")
            result["Missing Columns"].append(key)
    return result


def make_schema_getter(session, db):
    """
    Function factory to avoid late binding the db variable.
    """

    def f(no_llm: bool = False):
        if no_llm:
            return db
        # query table metadata
        column_metas = (
            session.query(TableMetadata)
            .filter(TableMetadata.id == db.connection_id)
            .all()
        )
        unique_tables_set = set()
        db_connector = connector_from_db_name(db.db_name)
        return f"sql_dialect: {db.sql_dialect}\n\nAbout Database: {db.instructions}\n\nInstructions: {db_connector.get_instructions(column_metas, unique_tables_set)}"

    return f


def make_sql_executor(db: DatabaseConnectionRead, return_str: bool = True):
    """
    Function factory to avoid late binding the db variable.
    """

    def f(sql: str):
        try:
            if return_str:
                res = query_database_connection(sql, db.db_name)
                return str(res)
            else:
                res = query_database_connection(sql, db.db_name, as_dict=True)
                return res
        except Exception as e:
            logging.error(e)
            return str(e)

    return f


def get_db_schema_tools(
    session: Session, database_connections
) -> List[StructuredTool]:
    selected_database_connections = database_connections
    return [
        StructuredTool.from_function(
            func=make_schema_getter(session, db),
            # The name must not have any white space
            name=f"{camel_case(db.connection_name)}SchemaGetter",
            description=f"useful for when you need to understand the schema of {db.connection_name} database",
        )
        for db in selected_database_connections
    ]


def get_sql_execution_tools(
    database_connections: List[DatabaseConnectionRead], return_str: bool = True
) -> List[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=make_sql_executor(db, return_str),
            # The name must not have any white space
            name=f"{camel_case(db.connection_name)}QueryExecutor",
            description=f"useful for when you need to execute SQL for {db.connection_name} database",
        )
        for db in database_connections
    ]


def camel_case(s):
    s = sub(r"([_\-])+", " ", s).title().replace(" ", "")
    # Join the string, ensuring the first letter is lowercase
    return "".join([s[0].lower(), s[1:]])


def extract_schema_getter(input_string):
    match = search(r"\b(\w*)SchemaGetter\b", input_string)
    if match:
        return match.group(1)
    return None


def get_connection_id_from_tools(schema_tool_name, tools):
    if not schema_tool_name or schema_tool_name == "":
        return None
    schema_tool = next((tool for tool in tools if tool.name == schema_tool_name), None)
    if not schema_tool:
        return None
    # get the connection id from the schema tool
    db = schema_tool.run({"no_llm": True})
    return db.connection_id if db else None
