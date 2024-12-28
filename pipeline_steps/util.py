import csv
import io
import logging
import re
from contextlib import contextmanager
from time import time
from typing import List, Dict, Any, TypeVar

from langchain_core.output_parsers import JsonOutputParser
from pydantic.v1 import BaseModel

parser = JsonOutputParser()


T = TypeVar("T", bound=BaseModel)


@contextmanager
def performance_monitor(endpoint_name: str):
    start_time = time()
    try:
        yield
    finally:
        execution_time = time() - start_time
        logging.info(f"{endpoint_name} executed in {execution_time:.2f} seconds")


def json_array_to_csv(json_array: List[Dict[str, Any]]) -> str:
    """
    Convert a JSON array to CSV format.

    Args:
    json_array (List[Dict[str, Any]]): A list of dictionaries representing the JSON array.

    Returns:
    str: CSV formatted string.
    """
    if not json_array or not isinstance(json_array, list):
        raise ValueError("Input must be a non-empty JSON array (list of dictionaries)")

    json_array = clean_dict_keys_for_sql(json_array)

    # Get the fieldnames from the first item in the JSON array
    fieldnames = list(json_array[0].keys())

    # Use StringIO to create an in-memory file-like object
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)

    # Write the header
    writer.writeheader()

    # Write the rows
    for item in json_array:
        writer.writerow(item)

    # Get the CSV data as a string
    csv_data = output.getvalue()
    output.close()

    return csv_data


def clean_dict_keys_for_sql(dict_list):
    def clean_key(key):
        if not key:
            return key
        # Convert to lowercase
        cleaned = key.lower()

        # Replace spaces, dots, and other special characters with underscores
        cleaned = re.sub(r"[^\w]+", "_", cleaned)

        # Replace 'ü' with 'u', 'ä' with 'a', 'ö' with 'o', 'ß' with 'ss'
        cleaned = (
            cleaned.replace("ü", "u")
            .replace("ä", "a")
            .replace("ö", "o")
            .replace("ß", "ss")
        )

        # Remove leading digits
        cleaned = re.sub(r"^\d+", "", cleaned)

        # Ensure the key starts with a letter or underscore
        if not cleaned[0].isalpha() and cleaned[0] != "_":
            cleaned = "_" + cleaned

        # Truncate to 64 characters (common SQL identifier length limit)
        cleaned = cleaned[:64]

        return cleaned

    # List of common SQL reserved words
    sql_reserved_words = {
        "select",
        "from",
        "where",
        "and",
        "or",
        "insert",
        "update",
        "delete",
        "table",
        "column",
        "database",
        "order",
        "by",
        "group",
        "having",
        "limit",
        "join",
        "left",
        "right",
        "inner",
        "outer",
        "on",
        "as",
        "distinct",
        "case",
        "when",
        "then",
        "else",
        "end",
        "union",
        "all",
        "into",
        "values",
        "set",
    }

    cleaned_dict_list = []
    for d in dict_list:
        cleaned_dict = {}
        for key, value in d.items():
            cleaned_key = clean_key(key)

            # Check if the key is a reserved word
            if cleaned_key.lower() in sql_reserved_words:
                cleaned_key = f'"{cleaned_key}"'

            # Handle potential duplicates
            count = 1
            original_key = cleaned_key
            while cleaned_key in cleaned_dict:
                cleaned_key = f"{original_key}_{count}"
                count += 1

            cleaned_dict[cleaned_key] = value

        cleaned_dict_list.append(cleaned_dict)

    return cleaned_dict_list


def sql_data_to_csv(data):
    """
    data: dictionary with keys 'columns' and 'rows'
    columns: list of dictionaries with keys 'column_name' and 'data_type'
    rows: list of dictionaries with keys matching column names
    """

    # Extract column names
    columns = [col["column_name"] for col in data["columns"]]
    csv_data = ",".join(columns) + "\n"

    # Extract 10 rows to avoid max token occurred error
    for row in data["rows"][:10]:
        csv_data += ",".join([str(row[col]) for col in columns]) + "\n"
    return csv_data


def tuple_to_csv(data):
    """
    data: list of tuples
    """
    csv_data = ""
    for row in data[:10]:
        csv_data += ",".join([str(val) for val in row]) + "\n"

    return csv_data
