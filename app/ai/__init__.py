"""
AI Learning Blueprint
"""
from flask import Blueprint

ai_bp = Blueprint('ai', __name__, template_folder='templates')

from . import routes
from . import chatbot

