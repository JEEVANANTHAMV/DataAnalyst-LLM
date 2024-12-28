import logging

from sqlalchemy import create_engine, Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker


db_user = ""
db_password = ""
db_name = ""
db_host = ""

DATABASE_URL = (
    f"postgresql://{db_user}:{db_password}@{db_host}/{db_name}?sslmode=require"
)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

CLIENT_DATABASE_CONNECTIONS_POOL = dict()


def get_connection_pool(db_connection: str) -> Engine:
    if not CLIENT_DATABASE_CONNECTIONS_POOL.get(db_connection):
        CLIENT_DATABASE_CONNECTIONS_POOL[db_connection] = create_engine(db_connection)
    return CLIENT_DATABASE_CONNECTIONS_POOL[db_connection]


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_session_local():
    return SessionLocal()


def get_engine_local():
    return engine


def commit_session(session):
    try:
        session.commit()
    except SQLAlchemyError as e:
        logging.error(e)
        session.rollback()
        raise e
