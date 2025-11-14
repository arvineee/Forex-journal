from flask import Blueprint

trades_bp = Blueprint('trades', __name__)

from app.trades import routes
