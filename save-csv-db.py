import pandas as pd
from connectdb import connect_to_postgresql, execute_query
from sqlalchemy import create_engine
data = pd.read_csv('dataLawFinal.csv')
df = pd.DataFrame(data)
print(df)

conn = connect_to_postgresql()

# cursor = conn.cursor()

# cursor.execute('''
#         CREATE TABLE IF NOT EXISTS qas (
#         id serial PRIMARY KEY,
#         title text NOT NULL,
#         embedding_title text not null,
#         content text not null,
#         source varchar(255)
#         )
#         ''')

# for row in df.itertuples():
#     cursor.execute('''
#         INSERT INTO qas (title, embedding_title, content, source)
#         VALUES (%s, %s, %s, %s)
#         ''',
#         (row.title, row.embedding_title, row.content, row.source))

# conn.commit()

query = '''SELECT * FROM qas'''
data = execute_query(connection=conn, query=query)
cols = [
    'id',
    'title',
    'embedding_title',
    'content',
    'source'
]
df = pd.DataFrame(data, columns=cols)
print(df.columns)