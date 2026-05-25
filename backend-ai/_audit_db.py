import sys; sys.path.insert(0,'.'); sys.stdout.reconfigure(encoding='utf-8')
from database.db import database

tables = database.run_query_df("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE' ORDER BY TABLE_NAME")
for _, r in tables.iterrows():
    tn = r['TABLE_NAME']
    try:
        cols = database.run_query_df(f"SELECT TOP 0 * FROM [{tn}]")
        cnt = database.run_scalar(f"SELECT COUNT(1) FROM [{tn}]")
        print(f"{tn} ({cnt} rows): {list(cols.columns)}")
    except Exception as e:
        print(f"{tn}: ERROR {e}")
