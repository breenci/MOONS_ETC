from ensurepip import bootstrap
from flask import Flask
from config import Config
from flask_bootstrap import Bootstrap
from flask_session import Session

app = Flask(__name__)
app.config.from_object(Config)
bootstrap = Bootstrap(app)
Session(app)

from app import routes