import os
from flask import Flask
from flask_mysqldb import MySQL
import sqlite3

print(__file__)
os.path.dirname(__file__)
db_name = "system_securit.db"
database_path = os.path.join(os.path.dirname(__file__),  db_name)


def create_first_database():
    db = sqlite3.connect(database_path)
    cr = db.cursor()
    cr.execute(
        """CREATE TABLE IF NOT EXISTS  data_set_lotte (
                id INTEGER NOT NULL PRIMARY KEY,
                documents_id INTEGER,
                documents TEXT
                )"""
    )
    cr.execute(
        """CREATE TABLE IF NOT EXISTS  data_set_antique (
                id INTEGER NOT NULL PRIMARY KEY,
                documents_id INTEGER,
                documents TEXT
                )"""
    )
    db.commit()
    db.close()

def create_app():
    app = Flask(__name__)
    # Configure MySQL settings
    # app.config['MYSQL_HOST'] = 'localhost'  # MySQL server hostname
    # app.config['MYSQL_USER'] = 'root'       # MySQL username
    # app.config['MYSQL_PASSWORD'] = ''  # MySQL password
    # app.config['MYSQL_DB'] = 'ir_database'        # Name of your database
    # app.config["SECRET_KEY"] = "cairocoders-ednalan"
    app.config["SECRET_KEY"] = "cairocoders-ednalan"
    create_first_database()
    # mysql = MySQL(app)
    return app
