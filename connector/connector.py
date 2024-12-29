import json
import base64
from datetime import datetime
from typing import Optional
from enum import Enum

from fastapi import HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Boolean,
    JSON,
    DateTime,
    func,
)
from sqlalchemy.orm import declarative_base
from cuid2 import cuid_wrapper
from .wrapper import PostgresWrapper
from typing import Callable

db_id_generator: Callable[[], str] = cuid_wrapper()

# Declarative base for SQLAlchemy models
DBModel = declarative_base()

#TODO - u
host = ""
port = ""
user = ""
password = ""
db_name = ""


class SupportDwhDialect(Enum):
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MS_SQL_SERVER = "mssql"

    @classmethod
    def supported_dialects(cls):
        return [dialect.value for dialect in cls]

    @classmethod
    def protocol(cls, dialect: str):
        return {
            cls.MYSQL.value: "mysql+mysqlconnector",
            cls.POSTGRESQL.value: "postgresql",
            cls.MS_SQL_SERVER.value: "mssql+pymssql",
        }.get(dialect, None)

    @classmethod
    def from_protocol(cls, protocol: str):
        return {
            "mysql+mysqlconnector": cls.MYSQL,
            "postgresql": cls.POSTGRESQL,
            "mssql+pymssql": cls.MS_SQL_SERVER,
        }.get(protocol, None)


def get_dwh_connector(sql_dialect, connect_str):
    if sql_dialect not in SupportDwhDialect.supported_dialects():
        raise HTTPException(status_code=400, detail="Unsupported dialect")

    if sql_dialect == SupportDwhDialect.POSTGRESQL.value:
        return PostgresWrapper(connect_str)
    else:
        raise HTTPException(status_code=400, detail="Unsupported dialect")


class DatabaseConnectionCreate(BaseModel):
    connection_name: str
    host: Optional[str] = None
    port: Optional[int] = None
    db_name: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    force_non_read_only: bool = Field(False)
    sql_dialect: str
    instructions: Optional[str] = None
    time_created: Optional[datetime] = None
    time_updated: Optional[datetime] = None


class DatabaseConnectionRead(DatabaseConnectionCreate):
    connection_id: str = Field(...)
    schema_ddl: Optional[str] = Field(...)
    is_owner: bool = False
    is_team_visible: bool = False

    class Config:
        populate_by_name = True
        json_encoders = {bytes: lambda x: x.decode()}
        from_attributes = True

    def __repr__(self):
        return f"{SupportDwhDialect.protocol(self.sql_dialect)}://{self.username}:{self.password}@{self.host}/{self.db_name}"

class DatabaseConnection(DBModel):
    __tablename__ = "database_connections"
    connection_id = Column(String(36), primary_key=True, default=db_id_generator())
    user_id = Column(String(36), nullable=True)
    connection_name = Column(String(255), nullable=False)
    host = Column(String(255), nullable=True)
    port = Column(Integer, nullable=True)
    db_name = Column(String(255), nullable=False)
    username = Column(String(255), nullable=True)
    password = Column(String(255), nullable=True)  # TODO: Store securely
    gcp_service_account = Column(JSON, nullable=True)
    force_non_read_only = Column(Boolean, default=False)
    schema_ddl = Column(Text, nullable=True)
    sql_dialect = Column(String(36), nullable=True)
    instructions = Column(Text, nullable=True)
    time_created = Column(DateTime, default=func.current_timestamp(), nullable=True)
    time_updated = Column(
        DateTime,
        default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
        nullable=True,
    )

    def __repr__(self):
        if self.sql_dialect == SupportDwhDialect.BIG_QUERY.value:
            gcp_data_string = json.dumps(self.gcp_service_account)
            data = base64.b64encode(gcp_data_string.encode("utf-8")).decode("utf-8")
            return f"{SupportDwhDialect.protocol(self.sql_dialect)}://{self.gcp_service_account.get('project_id')}/{self.db_name}?credentials_base64={data}"
        else:
            return f"{SupportDwhDialect.protocol(self.sql_dialect)}://{self.username}:{self.password}@{self.host}/{self.db_name}"


class TableMetadataCreate(BaseModel):
    id: str = Field(...)
    table_name: str = Field(..., max_length=255)
    column_name: str = Field(..., max_length=255)
    data_type: str = Field(...)
    column_description: Optional[str] = Field(None, alias="columnDescription")
    table_description: Optional[str] = Field(None, alias="tableDescription")
    time_created: Optional[datetime] = None
    time_updated: Optional[datetime] = None

class TableMetadataRead(TableMetadataCreate):
    metadata_id: str = Field(...)

    class Config:
        populate_by_name = True
        from_attributes = True
        json_encoders = {bytes: lambda x: x.decode()}
