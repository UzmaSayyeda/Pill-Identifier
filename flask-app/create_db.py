# import libraries
import sqlite3
import pandas as pd

# connect to database
conn = sqlite3.connect("drugs.db")
cursor = conn.cursor()

# read csv to pd.DataFrame
df = pd.read_csv('drug_side_effects.csv')
print(df.sample(10))

# add DataFrame to sql
df.to_sql("drug_effects", conn, if_exists="replace", index=False)
