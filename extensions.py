from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail
from flask import Flask

app = Flask(__name__)
db = SQLAlchemy()
mail = Mail()
