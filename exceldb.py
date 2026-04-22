import pandas as pd
import psycopg2

conn = psycopg2.connect(
    dbname="sujeet_db",
    user="fastapi_user",
    password="Sujeet123",
    host="localhost"
)

df = pd.read_sql("SELECT * FROM pdf_analyse_logs", conn)
df.to_excel("table_data.xlsx", index=False)