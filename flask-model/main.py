from flask import Flask, render_template, jsonify
import sqlalchemy as sql
from sqlalchemy import create_engine
import pandas as pd
import sqlite3

app = Flask(__name__)

db_url = 'sqlite:///drugs.db'
engine = sql.create_engine(db_url)

@app.route("/")
def index():
    query = "SELECT * FROM drug_effects"
    data = pd.read_sql(query, engine)
    drug_data = data.to_dict(orient='records')
    return render_template("index.html", drug_data=drug_data)

@app.route("/data")
def get_data():
    conn = sqlite3.connect("drugs.db")
    cursor = conn.cursor()
    query = "SELECT * FROM drug_effects"
    data_two = cursor.execute(query).fetchall()


    drug_data_two = []
    for row in data_two:
        row_dict = {}
        for i, column in enumerate(cursor.description):
            row_dict[column[0]] = row[i]
        drug_data_two.append(row_dict)

    conn.close()
    return jsonify(drug_data_two)

if __name__ == "__main__":
    app.run(debug=True)