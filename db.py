import logging

from sqlalchemy import create_engine, Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker


DB_HOST='35.198.119.144'
DB_PORT=5432
DB_USER='jeeva'
DB_PASSWORD='E30q5#mTfsKl19'
DB_NAME='jeeva_db'

DATABASE_URL = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}?sslmode=require"
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
