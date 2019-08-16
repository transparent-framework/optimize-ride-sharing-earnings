from sqlalchemy import *

class DBUtils:
    # Database access constants for mysql connections (Sensitive data hidden)

    database_server = 'xxxx'
    database_port = xxxx
    database_name = 'xxxx'
    database_username = 'xxxx'
    database_password = 'xxxx'

def get_db_connection():
    # Create SQLAlchemy engine and connection
    engine = create_engine ("mysql://{0}:{1}@{2}:{3}/{4}".format(
                                    DBUtils.database_username,
                                    DBUtils.database_password,
                                    DBUtils.database_server,
                                    DBUtils.database_port,
                                    DBUtils.database_name), encoding="utf-8")
    conn = engine.connect()
    return conn
