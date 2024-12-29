import enum
from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field

class DataSourceResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for the data source")
    name: str = Field(..., description="Name of the data source")
    database_name: str = Field(
        ..., description="Name of the database in the data warehouse"
    )
    is_active: bool = Field(..., description="Indicates if the data source is active")
    time_created: datetime = Field(
        ..., description="Timestamp when the data source was created"
    )
    time_updated: datetime = Field(
        ..., description="Timestamp when the data source was last updated"
    )