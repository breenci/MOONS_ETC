import os

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    SESSION_PERMANENT = False
    SESSION_TYPE = 'filesystem'
    SESSION_FILE_THRESHOLD = 2