import os

import pandas as pd
import streamlit as st
import psycopg2
from dotenv import load_dotenv
from langchain_chroma import Chroma
from brain import get_embedding
from functools import lru_cache

load_dotenv()

# @lru_cache(maxsize=1)
def connect_to_postgresql():
    try:
        connection = psycopg2.connect(
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_DATABASE")
        )
        return connection
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL: {e}")
        # st.error(f"Error connecting to PostgreSQL: {e}")
        return None

def execute_query(connection, query=""):
    if connection is None:
        raise Exception("PostgreSQL connection is not established.")
    cursor = connection.cursor()
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()
    return data

# @lru_cache(maxsize=1)
# def query_qas():
#     query = '''SELECT * FROM qas'''
#     connection = connect_to_postgresql()
#     data = execute_query(connection, query=query)
#     connection.close()
#     cols = [
#         'id',
#         'title',
#         'embedding_title',
#         'content',
#         'source'
#     ]
#     df = pd.DataFrame(data, columns=cols)
#     return df

def query_qas():
    """Fetch only the necessary columns from the database."""
    query = '''SELECT id, embedding_title FROM qas'''
    # query = '''SELECT * FROM qas'''
    connection = connect_to_postgresql()
    data = execute_query(connection, query=query)
    connection.close()
    return pd.DataFrame(data, columns=['id', 'embedding_title'])

def query_qas_details(ids):
    """Fetch detailed data for the given IDs."""
    query = f"SELECT id, title, content, source FROM qas WHERE id IN ({','.join(map(str, ids))})"
    connection = connect_to_postgresql()
    data = execute_query(connection, query=query)
    connection.close()
    return pd.DataFrame(data, columns=['id', 'title', 'content', 'source'])


def load_chroma():
    try:
        collection = Chroma(persist_directory="./chromadb",
                        embedding_function=get_embedding(),
                        collection_name="rag")
    except Exception as e:
        print(f"Error connecting to Chroma: {e}")
        # st.error(f"Error connecting to Chroma: {e}")
        return None
    return collection