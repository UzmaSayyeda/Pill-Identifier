from flask import Flask, render_template, jsonify
import sqlalchemy as sql
import sqlite3

app = Flask(__name__)

db_url = 'sqlite:///drugs.db'
engine = sql.create_engine(db_url)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data")
def get_data():
    
    conn = sqlite3.connect("drugs.db")
    cursor = conn.cursor()
    query = "SELECT * FROM drug_effects"
    data = cursor.execute(query).fetchall()
    conn.close()
    return jsonify(data)
    

if __name__ == "__main__":
    app.run(debug=True)