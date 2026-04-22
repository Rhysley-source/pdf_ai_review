import pandas as pd
import psycopg2

conn = psycopg2.connect(
    dbname="sujeet_db",
    user="fastapi_user",
    password="Sujeet123",
    host="localhost"
)

df = pd.read_sql("SELECT * FROM pdf_analyse_logs", conn)

# 🔥 Remove timezone from all datetime columns
for col in df.select_dtypes(include=["datetime64[ns, UTC]"]):
    df[col] = df[col].dt.tz_localize(None)

# Save to Excel
df.to_excel("table_data.xlsx", index=False)

print("✅ Exported successfully")