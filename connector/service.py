import logging
import re
from typing import List

from fastapi import HTTPException
from sqlalchemy import (
    Column,
    String,
    DateTime,
    func,
    Boolean,
    create_engine,
    text,
    Text,
    ForeignKey,
    MetaData,
)
import uuid
from datetime import timedelta, datetime
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, Session
from typing_extensions import Optional

from connector.connector import DatabaseConnectionRead
from connector.connector import get_dwh_connector
from . import schemas, get_dwh_dialect

DBModel = declarative_base()

#TODO - connecting to my database as command
DATA_WAREHOUSE = ""


class DataImport(DBModel):
    """
    Represents a data import to the data warehouse through data lake (GCS).
    """

    __tablename__ = "data_imports"
    id = Column(String(36), primary_key=True, default=lambda: uuid.uuid4())
    name = Column(String, unique=True, nullable=False)
    description = Column(String)
    source_type = Column(String(36), nullable=False)
    source_subtype = Column(String, nullable=False)
    gcs_path = Column(String)
    # TODO track additional meta data such as airbyte source config, airbyte connection id
    status = Column(String(36), default="pending")
    error_details = Column(Text, nullable=True)
    time_created = Column(DateTime, default=func.current_timestamp(), nullable=True)
    time_updated = Column(
        DateTime,
        default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
        nullable=True,
    )


def get_data_import(db: Session, import_id: str) -> schemas.DataImportResponse:
    data_import = db.query(DataImport).filter(DataImport.id == import_id).first()
    if data_import is None:
        raise HTTPException(status_code=404, detail="Data import not found")
    return schemas.DataImportResponse(
        id=data_import.id,
        name=data_import.name,
        description=data_import.description,
        source_type=data_import.source_type,
        source_subtype=data_import.source_subtype,
        gcs_path=data_import.gcs_path,
        status=data_import.status,
        error_details=data_import.error_details,
        time_created=data_import.time_created,
        time_updated=data_import.time_updated,
    )


def list_data_imports(
    db: Session, page: int = 1, size: int = 10, filter: Optional[bool] = False
) -> schemas.PaginatedDataImportResponse:
    """List data imports with pagination.

    Parameters:
    - db: Database session
    - page: Page number (default: 1)
    - size: Number of items per page (default: 10)

    Returns:
    - PaginatedDataImportResponse: Paginated list of data imports
    """
    total = db.query(DataImport).count()
    pages = (total + size - 1) // size
    one_day_ago = datetime.utcnow() - timedelta(days=1)
    if filter:
        data_imports = (
            db.query(DataImport)
            .filter(
                DataImport.status == "ready",  # Filter by status 'ready'
                DataImport.time_created
                < one_day_ago,  # Filter for imports older than 1 day
            )
            .order_by(DataImport.time_created.desc())
            .offset((page - 1) * size)
            .limit(size)
            .all()
        )
    else:
        data_imports = (
            db.query(DataImport)
            .order_by(DataImport.time_created.desc())
            .offset((page - 1) * size)
            .limit(size)
            .all()
        )

    return schemas.PaginatedDataImportResponse(
        items=[
            schemas.DataImportResponse(
                id=di.id,
                name=di.name,
                description=di.description,
                source_type=di.source_type,
                source_subtype=di.source_subtype,
                gcs_path=di.gcs_path,
                status=di.status,
                error_details=di.error_details,
                time_created=di.time_created,
                time_updated=di.time_updated,
            )
            for di in data_imports
        ],
        total=total,
        page=page,
        size=size,
        pages=pages,
    )


class DataSource(DBModel):
    """
    Represents a data source in the system.

    The database_name field stores the name of the database to connect to.
    The actual connection to the data warehouse is handled via the DWH_MSSQL environment variable.
    """

    __tablename__ = "data_sources"
    id = Column(String(36), primary_key=True, default=lambda: uuid.uuid4())
    name = Column(String, unique=True, nullable=False)
    database_name = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    time_created = Column(DateTime, default=func.current_timestamp(), nullable=True)
    time_updated = Column(
        DateTime,
        default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
        nullable=True,
    )


class TableMetadata(DBModel):
    __tablename__ = "database_table_metadata"
    metadata_id = Column(
        String(36), primary_key=True, default=lambda: uuid.uuid4()
    )
    id = Column(String(36), ForeignKey("data_sources.id"))
    table_name = Column(String(255), nullable=False)
    column_name = Column(String(255), nullable=False)
    data_type = Column(Text, nullable=False)
    column_description = Column(Text)
    table_description = Column(Text)
    time_created = Column(DateTime, default=func.current_timestamp(), nullable=True)
    time_updated = Column(
        DateTime,
        default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
        nullable=True,
    )


def create_database_metadata(db: Session, data_source: DataSource):
    """
    TODO move to dwh_connectors
    Reads the `data_sources` table to get the list of database names and updates
    or inserts metadata for each table in the `database_table_metadata` table.

    :param db: SQLAlchemy Session object
    """
    # Get the list of database names from the `data_sources` table
    metadata_table = (
        db.query(TableMetadata).filter(TableMetadata.id == data_source.id).all()
    )
    if metadata_table:
        return
    engine = get_dwh_engine(data_source.database_name)
    metadata = MetaData()
    metadata.reflect(bind=engine)
    for table in metadata.sorted_tables:
        table_desc = ""
        table_name = table.name
        for column in table.columns:
            column_type = str(column.type)
            column_desc = ""
            column_name = column.name
            new_col_metadata = TableMetadata(
                id=data_source.id,
                table_name=table_name,
                column_name=column_name,
                data_type=column_type,
                column_description=column_desc,
                table_description=table_desc,
            )
            db.add(new_col_metadata)
            db.commit()
            db.refresh(new_col_metadata)


def create_data_source(
    db: Session, data_source: schemas.DataSourceCreate
) -> schemas.DataSourceResponse:
    if not validate_database_exists(data_source.database_name):
        raise ValueError("Database does not exist in the data warehouse")

    new_data_source = DataSource(
        name=data_source.name,
        database_name=data_source.database_name,
    )
    db.add(new_data_source)
    db.commit()
    db.refresh(new_data_source)
    return schemas.DataSourceResponse(
        id=new_data_source.id,
        name=new_data_source.name,
        database_name=new_data_source.database_name,
        is_active=new_data_source.is_active,
        time_created=new_data_source.time_created,
        time_updated=new_data_source.time_updated,
    )


def get_data_source(db: Session, data_source_id: str) -> schemas.DataSourceResponse:
    data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
    if data_source is None:
        raise HTTPException(status_code=404, detail="Data source not found")
    return schemas.DataSourceResponse(
        id=data_source.id,
        name=data_source.name,
        database_name=data_source.database_name,
        is_active=data_source.is_active,
        time_created=data_source.time_created,
        time_updated=data_source.time_updated,
    )


def update_data_source(
    db: Session, data_source_id: str, data_source: schemas.DataSourceUpdate
) -> schemas.DataSourceResponse:
    db_data_source = (
        db.query(DataSource).filter(DataSource.id == data_source_id).first()
    )
    if db_data_source is None:
        raise HTTPException(status_code=404, detail="Data source not found")

    if (
        data_source.database_name
        and data_source.database_name != db_data_source.database_name
    ):
        if not validate_database_exists(data_source.database_name):
            raise HTTPException(
                status_code=400, detail="Database does not exist in the data warehouse"
            )

    for key, value in data_source.dict(exclude_unset=True).items():
        setattr(db_data_source, key, value)

    db.commit()
    db.refresh(db_data_source)
    return schemas.DataSourceResponse(
        id=db_data_source.id,
        name=db_data_source.name,
        database_name=db_data_source.database_name,
        is_active=db_data_source.is_active,
        time_created=db_data_source.time_created,
        time_updated=db_data_source.time_updated,
    )


def delete_data_source(db: Session, data_source_id: str) -> bool:
    data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
    if data_source is None:
        raise HTTPException(status_code=404, detail="Data source not found")
    db.delete(data_source)
    db.commit()
    return True


def list_data_sources(db: Session) -> List[schemas.DataSourceResponse]:
    data_sources = db.query(DataSource).all()
    return [
        schemas.DataSourceResponse(
            id=ds.id,
            name=ds.name,
            database_name=ds.database_name,
            is_active=ds.is_active,
            time_created=ds.time_created,
            time_updated=ds.time_updated,
        )
        for ds in data_sources
    ]


def validate_database_exists(database_name: str) -> bool:
    connection_string = DATA_WAREHOUSE
    if not connection_string:
        raise ValueError("DWH_MSSQL environment variable is not set")

    try:
        engine = create_engine(connection_string)
        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT 1 FROM sys.databases WHERE name = :name"),
                {"name": database_name},
            )
            return result.fetchone() is not None
    except SQLAlchemyError as e:
        logging.error(f"Error connecting to the data warehouse: {str(e)}")
        return False


def get_source_tables(data_source_id, session: Session, hide_empty: Optional[bool]):
    source = session.query(DataSource).filter(DataSource.id == data_source_id).first()
    if not source:
        raise HTTPException(status_code=404, detail="Data source not found")

    connection_string = f"{DATA_WAREHOUSE}/{source.database_name}"
    con = connector(connection_string)
    tables = sorted(con.get_database_tables())

    if hide_empty:
        tables = [
            table for table in tables if con.get_database_table_row_count(table) > 0
        ]
    return tables


def get_source_table_data(
    data_source_id, table_name, page, limit, search, sort, filter, session: Session
):
    """
    data_source_id: str
    table_name: str
    page: int
    limit: int
    search: str or None Ex: col_01:value_01,col_02:value_02
    sort: str or None Ex: col_01:1,col_02:-1
    filter_data: str or None Ex: col_01:value_01,col_02:value_02
    """

    source = session.query(DataSource).filter(DataSource.id == data_source_id).first()
    if not source:
        raise HTTPException(status_code=404, detail="Database not found")

    connection_string = f"{DATA_WAREHOUSE}/{source.database_name}"
    con = connector(connection_string)

    return con.get_database_table_rows(table_name, page, limit, sort, filter, search)


def is_valid_select_query(query_string):
    if not isinstance(query_string, str) or len(query_string) < 10:
        return False

    # This regex aims to find disallowed keywords outside of quoted strings
    disallowed_keywords = r"\b(DELETE|INSERT|UPDATE|TRUNCATE|CREATE|ALTER|DROP)\b"
    pattern = re.compile(disallowed_keywords, re.IGNORECASE)

    # Split the query by single or double quotes to exclude contents within quotes
    tokens = re.split(
        r"""(?x)(?:  # verbose regex
        " (?: \\. | [^"] )* "   |   # double quoted items
        ' (?: \\. | [^'] )* '       # single quoted items
    ) | \s+                        # spaces
    """,
        query_string,
    )

    # Check only non-quoted parts for disallowed keywords
    for token in tokens:
        if pattern.search(token) and not token.startswith(("'", '"')):
            return False

    # Check if it starts with SELECT or WITH
    if re.match(r"^\s*(SELECT|WITH)\s", query_string, re.IGNORECASE):
        return True

    return False


def execute_custom_query(database_id, query, page, limit, filters, session: Session):
    if not is_valid_select_query(query):
        raise HTTPException(
            status_code=400, detail="Only valid 'SELECT' query supported"
        )

    source = session.query(DataSource).filter(DataSource.id == database_id).first()
    if not source:
        raise HTTPException(status_code=404, detail="Database not found")

    connection_string = f"{DATA_WAREHOUSE}/{source.database_name}"
    con = connector(connection_string)
    return con.execute_custom_query(query, page, limit, filters)


def create_datasource_for_import(
    database_name: str, data_import: DataImport, session: Session
) -> bool:
    # check if data source already exists
    if session.query(DataSource).filter(DataSource.name == data_import.name).first():
        logging.info(f"Data source {data_import.name} already exists")
        return True

    # create data source
    data_source = DataSource(
        name=data_import.name,
        database_name=database_name,
        is_active=True,
    )
    session.add(data_source)
    session.commit()
    logging.info(f"Created data source {data_import.name}")
    return True


def create_datasource_for_bq_import(
    dataset_id, data_import: DataImport, session: Session
) -> bool:
    # check if data source already exists
    if session.query(DataSource).filter(DataSource.name == data_import.name).first():
        logging.info(f"Data source {data_import.name} already exists")
        return True

    # create data source
    data_source = DataSource(
        name=data_import.name,
        database_name=dataset_id,
        is_active=True,
    )
    session.add(data_source)
    session.commit()
    logging.info(f"Created data source {data_import.name}")
    return True


def get_dwh_engine(database_name: str):
    """Get a connection to the data warehouse database"""
    connection_string = f"{DATA_WAREHOUSE}/{database_name}"
    con = connector(connection_string)
    return con.get_engine()


def connector_from_db_name(database_name: str):
    """Get a connection to the data warehouse database"""
    connection_string = f"{DATA_WAREHOUSE}/{database_name}"
    return connector(connection_string)


def connector(connection_string):
    return get_dwh_connector(get_dwh_dialect(), connection_string)


def query_database_connection(sql: str, database_name: str, as_dict: bool = False):
    db = get_dwh_engine(database_name)
    with db.connect() as c:
        res = c.execute(text(sql))
        if not as_dict:
            return res.fetchall()
        keys = res.keys()
        result_as_dict = [dict(zip(keys, row)) for row in res.fetchall()]
        return result_as_dict


def import_table_name(data_import_name: str) -> str:
    """Get the name of the database to import to"""
    table_name = re.sub(r"[^a-zA-Z0-9]", "_", data_import_name).lower()
    if not table_name:
        raise ValueError("Invalid database name")
    return f"{table_name}"


def get_connections_by_data_source(user_id: str, data_source_id: str, session: Session):
    data_source = session.query(DataSource).filter_by(id=data_source_id).first()
    if not data_source:
        raise HTTPException(status_code=404, detail="Data source not found")
    connections = []
    connection = DatabaseConnectionRead(
        connection_id=uuid.uuid4(),
        db_name=data_source.database_name,
        connection_name=data_source.name,
        sql_dialect=get_dwh_dialect(),
        schema_ddl=None,
        # TODO add an instructions field to the data source table
        instructions="General",
        gcp_service_account=None,
        is_owner=True,
        is_team_visible=True,
    )
    connections.append(connection)
    create_database_metadata(session, data_source)

    return connections


def get_database_tables(database_id, session):
    database = session.query(DataSource).filter_by(id=database_id).first()
    if not database:
        raise HTTPException(status_code=404, detail="Database not found")

    database_wrapper = connector(f"{DATA_WAREHOUSE}/{database.database_name}")

    return sorted(database_wrapper.get_database_tables())
