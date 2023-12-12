import sqlite3
import pandas as pd

conn = sqlite3.connect("drugs.db")
cursor = conn.cursor()


df = pd.read_csv('drug_side_effects.csv')
print(df.sample(10))

df.to_sql("drug_effects", conn, if_exists="replace", index=False)


        
