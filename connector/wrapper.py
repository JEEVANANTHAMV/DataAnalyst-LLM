import datetime
import decimal
import logging

import pandas as pd
from fastapi import HTTPException
from sqlalchemy import create_engine, text

from .base_wrapper import BaseWrapper

postgresql = """Given an input question, first create a syntactically correct postgresql query to run.

DO the following according to postgresql dialect.

Unless the user specifies in the question a specific number of examples to obtain. You should only order according to the distance function.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap 
each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do 
not exist. Also, pay attention to which column is in which table.
Pay attention to use today() function to get the current date, if the question involves "today". `ORDER BY` clause 
should always be after `WHERE` clause. DO NOT add semicolon to the end of SQL. Pay attention to the comments 
in the table schema. 
PAY attention to datatypes of columns and use explicit casting if needed. For example, if you
need to compare a column of type `date` with a string, you need to cast the date column to date.

For fuzzy string matching, use the pg_trgm extension's similarity function. The syntax is:
WHERE similarity(column, 'search_term') > 0.3
ORDER BY similarity(column, 'search_term') DESC
Adjust the threshold (0.3) as needed for more or less strict matching.
"""


class PostgresWrapper(BaseWrapper):
    def __init__(self, connection_string):
        self.instructions = postgresql
        self.engine = create_engine(connection_string)

    def get_llm_instructions(self):
        return self.instructions

    def get_engine(self):
        return self.engine

    def get_database_tables(self):
        try:
            tables = pd.read_sql(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'",
                con=self.engine,
            )
            return list(tables["table_name"])
        except Exception as e:
            logging.error(e)
            raise Exception("Failed to get database tables")

    def database_exists(self, database_name):
        try:
            with self.engine.connect().execution_options(
                isolation_level="AUTOCOMMIT"
            ) as connection:
                result = connection.execute(
                    text(f"SELECT 1 FROM pg_database WHERE datname='{database_name}'")
                )
                return bool(result.scalar())
        except Exception as e:
            logging.error(e)
            raise Exception("Failed to check if database exists")

    def create_database(self, database_name):
        try:
            with self.engine.connect().execution_options(
                isolation_level="AUTOCOMMIT"
            ) as connection:
                connection.execute(text(f"CREATE DATABASE {database_name}"))
        except Exception as e:
            logging.error(e)
            raise Exception("Failed to create database")

    def get_database_table_columns(self, table_name):
        try:
            tables = self.get_database_tables()

            if table_name not in tables:
                raise HTTPException(
                    status_code=404, detail=f"Table {table_name} not found"
                )

            columns = pd.read_sql(
                f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}'",
                con=self.engine,
            )
            return columns.to_dict(orient="records")
        except Exception as e:
            logging.error(e)
            raise Exception("Failed to get database table columns")

    def get_database_table_rows(
        self, table_name, page=1, limit=10, sort=None, filter=None, search=None
    ):
        """
        table_name: str Ex: users
        page: int Ex: 1
        limit: int Ex: 10
        sort: str Ex: col_01:1,col_02:-1
        filter: str Ex: col_01:value_01,col_02:value_02
        search: str Ex: col_01:value_01,col_02:value_02
        """
        try:
            tables = self.get_database_tables()
            if table_name not in tables:
                raise HTTPException(
                    status_code=404, detail=f"Table {table_name} not found"
                )

            columns = self.get_database_table_columns(table_name)
            int_columns = [
                column["column_name"]
                for column in columns
                if column["data_type"] == "integer"
            ]

            data_query = f'SELECT * FROM "{table_name}"'
            count_query = f'SELECT COUNT(*) FROM "{table_name}"'

            if filter:
                filter_fields = [
                    condition.split(":") for condition in filter.split(",")
                ]
                filter_conditions = [
                    (
                        f"{item[0]} = {int(item[1])}"
                        if item[0] in int_columns
                        else f"{item[0]} = '{item[1]}'"
                    )
                    for item in filter_fields
                ]
                data_query += " WHERE " + " AND ".join(filter_conditions)
                count_query += " WHERE " + " AND ".join(filter_conditions)

            if search:
                search_fields = [
                    condition.split(":") for condition in search.split(",")
                ]
                search_conditions = [
                    (
                        f"{item[0]} = {int(item[1])}"
                        if item[0] in int_columns
                        else f"{item[0]} ILIKE '%{item[1]}%'"
                    )
                    for item in search_fields
                ]

                if filter:
                    data_query += " AND " + " AND ".join(search_conditions)
                    count_query += " AND " + " AND ".join(search_conditions)
                else:
                    data_query += " WHERE " + " AND ".join(search_conditions)
                    count_query += " WHERE " + " AND ".join(search_conditions)

            if sort:
                sort_fields = [condition.split(":") for condition in sort.split(",")]
                sort_conditions = [
                    f"{item[0]} {'ASC' if int(item[1]) == 1 else 'DESC'}"
                    for item in sort_fields
                ]
                data_query += " ORDER BY " + ", ".join(sort_conditions)

            if page is not None and limit is not None:
                data_query += f" LIMIT {limit} OFFSET {(page - 1) * limit}"

            logging.info(data_query)
            with self.engine.connect() as connection:
                data = list(connection.execute(text(data_query)).mappings())
                count = list(connection.execute(text(count_query)).fetchone())[0]

            # FIX: search not working with pd.read_sql and pd.read_sql_query
            # data = pd.read_sql_query(data_query, con=self.engine).to_dict(orient='records')
            # count = pd.read_sql_query(count_query, con=self.engine).to_dict(orient='records')[0]['count']

            total_pages = (
                int(count / limit) + 1 if count % limit > 0 else int(count / limit)
            )

            return {
                "columns": columns,
                "rows": data,
                "total_pages": total_pages,
                "count": count,
            }
        except Exception as e:
            logging.error(e)
            raise Exception("Failed to get table data")

    def get_database_table_row_count(self, table_name):
        try:
            count_query = f"SELECT COUNT(*) FROM {table_name}"
            with self.engine.connect() as connection:
                count = list(connection.execute(text(count_query)).fetchone())[0]
            return count
        except Exception as e:
            logging.error(e)
            raise Exception("Failed to get row count")

    def column_type(self, value):
        try:
            if type(value) == str:
                return "text"
            if (
                type(value) == int
                or type(value) == float
                or type(value) == decimal.Decimal
            ):
                return "integer"
            if type(value) == datetime.datetime or type(value) == datetime.date:
                return "timestamp without timezone"
            return "text"

            # FIX: search not working with pd.read_sql and pd.read_sql_query
            # return pd.read_sql(query, con=self.engine).to_dict(orient='records')
        except Exception as e:
            logging.error(e)
            raise Exception("Failed to get column type")

    def create_where_condition(self, filter_obj):
        """
        :param filter_obj: A dictionary containing filter conditions
        :return: A string representing the SQL WHERE clause
        """
        try:
            conditions = []

            # Handle dropdown filters
            if "dropdown" in filter_obj:
                for column, values in filter_obj["dropdown"].items():
                    value = ", ".join([f"'{value}'" for value in values])
                    condition = f"{column} IN ({value})"
                    conditions.append(condition)

            # Handle datetime picker filters
            if "datetime_picker" in filter_obj:
                for column, datetime_range in filter_obj["datetime_picker"].items():
                    if "from" in datetime_range and "to" in datetime_range:
                        condition = f"{column} BETWEEN '{datetime_range['from']}' AND '{datetime_range['to']}'"
                    elif "from" in datetime_range:
                        condition = f"{column} >= '{datetime_range['from']}'"
                    elif "to" in datetime_range:
                        condition = f"{column} <= '{datetime_range['to']}'"
                    conditions.append(condition)

            # Handle numeric filters
            if "numeric" in filter_obj:
                for column, value in filter_obj["numeric"].items():
                    condition = f"{column} = {value}"
                    conditions.append(condition)

            # Handle text filters
            if "text" in filter_obj:
                for column, value in filter_obj["text"].items():
                    condition = f"{column} ILIKE '%{value}%'"
                    conditions.append(condition)

            # Combine all conditions with AND
            where_clause = " AND ".join(conditions)

            return where_clause
        except Exception as e:
            logging.error(e)
            raise Exception("Failed to create where condition")
