import logging
import re
from abc import ABC
from typing import List, Optional, Dict, Any
import sqlparse
import yaml
from sqlalchemy import MetaData, Engine, select, text

import re

def remove_sql_comments(sql_query: str):
    try:
        # Function to replace comments outside strings
        def replace_comments(match):
            if match.group(1):  # If it's a string literal, return it unchanged
                return match.group(0)
            else:  # Otherwise, it's a comment, so remove it
                return " "

        # Regex pattern to match string literals and comments
        pattern = r"('(?:''|[^'])*')|(--[^\n]*|/\*[\s\S]*?\*/|#[^\n]*)"

        # Replace comments while preserving string literals
        cleaned_query = re.sub(pattern, replace_comments, sql_query)

        # Remove any excess whitespace
        cleaned_query = re.sub(r"\s+", " ", cleaned_query).strip()

        return cleaned_query
    except Exception as e:
        print(f"Error occurred while removing comments: {e}")
        return sql_query



class ValidationError(Exception):
    def __init__(self, message: str, errors: List[str]):
        self.message = message
        self.errors = errors
        super().__init__(self.message)


class ProcessingError(Exception):
    pass


class BaseWrapper(ABC):
    allowed_privileges = ["SELECT", "USAGE"]

    def get_llm_instructions(self):
        pass

    def get_database_tables(self) -> List[str]:
        pass

    def get_database_table_columns(self, table_name):
        pass

    def get_database_table_rows(
        self, table_name, limit=None, offset=None, sort=None, filter=None, search=None
    ):
        pass

    def get_database_table_row_count(self, table_name):
        pass

    def create_where_condition(self, filters: dict):
        pass

    def get_engine(self) -> Engine:
        pass

    def create_database(self, database_name):
        pass

    def clean_name(self, name):
        return str(name)

    def database_exists(self, database_name):
        pass

    def compress_value(self, value, max_words=20):
        if isinstance(value, str):
            words = value.split()
            if len(words) > max_words:
                return " ".join(words[:max_words]) + "..."
        # elif isinstance(value, (int, float)):
        #     return round(value, 2)
        elif isinstance(value, list):
            return f"List({len(value)} items)"
        elif isinstance(value, bytes):
            return f"BLOB({len(value)} bytes)"
        elif isinstance(value, dict):
            return self.compress_jsonb(value)
        return ""

    def compress_jsonb(self, jsonb_value):
        # root_keys = list(jsonb_value.keys())
        return "Json field"

    def truncate_type(self, type_name: str, max_type_length=7) -> str:
        if len(type_name) > max_type_length:
            return type_name[:max_type_length]
        return type_name

    def get_instructions(self, column_meta: List, selected_tables: List[str] = None):
        """
        Generate instructions for the database schema and sample data.
        :param column_meta: table and column metadata based on context provided by the user
        :param selected_tables: selected tables by the user
        :return:
        """
        current_metadata = MetaData()
        current_metadata.reflect(bind=self.get_engine())

        schema_dict: Dict[str, Any] = {}
        sample_data = self.get_sample_rows()

        for table in current_metadata.sorted_tables:
            table_name = self.clean_name(table.name)
            if not selected_tables or table_name in selected_tables:
                schema_dict[table_name] = {"c": {}}

        for item in column_meta:
            table_name = self.clean_name(item.table_name)
            if not selected_tables or table_name in selected_tables:
                column_name = self.clean_name(item.column_name)
                if item.column_description:
                    column_type = self.truncate_type(
                        str(
                            current_metadata.tables[table_name]
                            .columns[column_name]
                            .type
                        )
                    )
                    schema_dict[table_name]["c"][column_name] = [
                        item.column_description,
                        None,
                        column_type,
                    ]

        for table_name, row in sample_data.items():
            table_name = self.clean_name(table_name)
            if not selected_tables or table_name in selected_tables:
                for column_name, value in row.items():
                    column_name = self.clean_name(column_name)
                    column_type = self.truncate_type(
                        str(
                            current_metadata.tables[table_name]
                            .columns[column_name]
                            .type
                        )
                    )
                    if column_name in schema_dict[table_name]["c"]:
                        compressed_value = self.compress_value(value)
                        schema_dict[table_name]["c"][column_name][1] = compressed_value
                    else:
                        schema_dict[table_name]["c"][column_name] = [
                            "",
                            self.compress_value(value),
                            column_type,
                        ]

        yaml_schema = yaml.dump(schema_dict, default_flow_style=False, sort_keys=False)

        instructions = f"{self.get_llm_instructions()}\n\nDatabase/Dataset Name:{self.get_engine().url.database} \nDB Schema Yaml (t=table,c=columns,[desc,sample]):\n{yaml_schema}"
        return instructions

    def get_sample_rows(self):
        # Reflect the database schema into a new metadata object
        try:
            metadata = MetaData()
            engine = self.get_engine()
            metadata.reflect(bind=engine)

            sample_rows = {}
            with engine.connect() as connection:
                for table in metadata.sorted_tables:
                    try:
                        # Select the first row from each table
                        query = select(table).limit(1)
                        result = connection.execute(query).fetchone()
                        if result:
                            # Combine column names with their values for clear output
                            sample_data = {
                                column.name: result[idx]
                                for idx, column in enumerate(table.columns)
                            }
                            sample_rows[table.name] = sample_data
                            logging.debug(f"Sample from {table.name}: {sample_data}")
                        else:
                            logging.warning(f"No data available in {table.name}")
                    except Exception as e:
                        logging.error(f"Failed to fetch from {table.name}: {e}")
            return sample_rows
        except Exception as e:
            logging.error(f"Failed to fetch sample rows: {e}")
            raise Exception("Failed to fetch sample rows")

    def split_query(self, query):
        try:
            # Initialize the removed clauses dictionary
            removed_clauses = {
                "group_by": "",
                "order_by": "",
                "limit": "",
                "offset": "",
                "where": "",
                "having": "",
            }

            if query.endswith(";"):
                query = query[:-1]

            # Extract and remove LIMIT clause
            limit_match = re.search(
                r"LIMIT (\d+)", query, flags=re.IGNORECASE | re.DOTALL
            )
            if limit_match:
                removed_clauses["limit"] = limit_match.group(1).strip()
                query = query.replace(limit_match.group(0), "")

            # Extract and remove OFFSET clause
            offset_match = re.search(
                r"OFFSET (\d+)", query, flags=re.IGNORECASE | re.DOTALL
            )
            if offset_match:
                removed_clauses["offset"] = offset_match.group(1).strip()
                query = query.replace(offset_match.group(0), "")

            # Extract and remove GROUP BY clause
            group_by_match = re.search(
                r"GROUP BY (.*?)(?=ORDER BY|HAVING|WHERE|$)",
                query,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if group_by_match:
                removed_clauses["group_by"] = group_by_match.group(1).strip()
                query = query.replace(group_by_match.group(0), "")

            # Extract and remove ORDER BY clause
            order_by_match = re.search(
                r"ORDER BY (.*?)(?=GROUP BY|HAVING|WHERE|$)",
                query,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if order_by_match:
                removed_clauses["order_by"] = order_by_match.group(1).strip()
                query = query.replace(order_by_match.group(0), "")

            # Extract and remove HAVING clause
            having_match = re.search(
                r"HAVING (.*?)(?=GROUP BY|ORDER BY|WHERE|$)",
                query,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if having_match:
                removed_clauses["having"] = having_match.group(1).strip()
                query = query.replace(having_match.group(0), "")

            # Extract and remove WHERE clause
            where_match = re.search(
                r"WHERE (.*?)(?=GROUP BY|ORDER BY|HAVING|LIMIT|OFFSET|$)",
                query,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if where_match:
                removed_clauses["where"] = where_match.group(1).strip()
                query = query.replace(where_match.group(0), "")

            # Strip any trailing whitespace
            query = query.strip()

            return query, removed_clauses
        except Exception as e:
            logging.error(f"Failed to split query: {e}")
            raise Exception("Failed to split query")

    def join_query(self, base_query: str, clauses: dict) -> str:
        # Reconstruct the query with the removed clauses
        try:
            query = base_query

            if clauses["where"]:
                query += f" WHERE {clauses['where']}"

            if clauses["group_by"]:
                query += f" GROUP BY {clauses['group_by']}"

            if clauses["having"]:
                query += f" HAVING {clauses['having']}"

            if clauses["order_by"]:
                query += f" ORDER BY {clauses['order_by']}"

            if clauses["limit"]:
                query += f" LIMIT {clauses['limit']}"

            if clauses["offset"]:
                query += f" OFFSET {clauses['offset']}"

            return query
        except Exception as e:
            logging.error(f"Failed to join query: {e}")
            raise Exception("Failed to join query")

    def update_query(
        self, query: str, condition: Optional[str], limit: int, offset: int
    ):
        """
        Assuming one query per function call
        """
        try:
            parsed = sqlparse.parse(query)[0]
            tokens = []

            where_found = False
            for token in parsed.tokens:
                if isinstance(token, sqlparse.sql.Where):
                    where_found = True
                    if condition:
                        tokens.append(token.value.strip() + " AND " + condition)
                    else:
                        tokens.append(token.value.strip())
                elif token.value.strip() == "":
                    pass
                else:
                    tokens.append(token.value.strip())

            if "LIMIT" in tokens:
                index = tokens.index("LIMIT")
                tokens[index + 1] = str(limit)
            elif "limit" in tokens:
                index = tokens.index("limit")
                tokens[index + 1] = str(limit)
            else:
                tokens.append(f" LIMIT {limit}")

            if "OFFSET" in tokens:
                index = tokens.index("OFFSET")
                tokens[index + 1] = str(offset)
            elif "offset" in tokens:
                index = tokens.index("offset")
                tokens[index + 1] = str(offset)
            else:
                tokens.append(f" OFFSET {offset}")

            if condition and not where_found:
                insert_pos = 0
                for i, token in enumerate(tokens):
                    if token.upper() in [
                        "GROUP BY",
                        "HAVING",
                        "ORDER BY",
                        "LIMIT",
                        "OFFSET",
                    ]:
                        break
                    insert_pos = i + 1

                new_where_clause = f" WHERE {condition}"
                tokens.insert(insert_pos, new_where_clause)

            return " ".join(tokens).replace("  ", " ").strip()
        except Exception as e:
            logging.error(f"Failed to extract limit and offset: {e}")
            raise Exception("Failed to extract limit and offset")

    def execute_custom_query(
        self, query: str, page=1, limit=10, filters= None
    ):
        try:
            if query[-1] == ";":
                query = query[:-1]

            query = remove_sql_comments(query)

            if page < 1 or limit < 1:
                raise ValueError("Page and limit must be positive integers.")

            with self.engine.connect() as connection:
                count = connection.execute(text(query)).rowcount

                max_limit, default_offset = self.get_limit_and_offset(query)
                if not max_limit:
                    max_limit = count

                if not default_offset:
                    default_offset = 0

                total_pages = (
                    int(count / limit) + 1 if count % limit > 0 else int(count / limit)
                )

                updated_limit, updated_offset = self.calculate_limit_and_offset(
                    page, total_pages, limit, max_limit, default_offset
                )

                if filters:
                    conditions = self.create_where_condition(
                        filters.model_dump(exclude_unset=True)
                    )
                else:
                    conditions = None

                updated_query = self.update_query(
                    query, conditions, updated_limit, updated_offset
                )

                logging.info(f"Executing query: {updated_query}")
                result_proxy = connection.execute(text(updated_query))
                data = list(result_proxy.mappings())

            if len(data) == 0:
                return {
                    "columns": [],
                    "rows": [],
                    "total_pages": total_pages,
                    "count": count,
                }

            columns = []
            for key, value in data[0].items():
                columns.append(
                    {"column_name": key, "data_type": self.column_type(value)}
                )

            return {
                "columns": columns,
                "rows": data,
                "total_pages": total_pages,
                "count": count,
            }
        except Exception as e:
            logging.error(f"Failed to execute query: {e}")
            raise Exception("Failed to execute query")


    def import_db_name(self, data_import_name: str) -> str:
        """Get the name of the database to import to"""
        db_name = re.sub(r"[^a-zA-Z0-9]", "_", data_import_name).lower()
        if not db_name:
            raise ValueError("Invalid database name")
        return f"imported_{db_name}"
