from flask import Blueprint

account_bp = Blueprint('account', __name__)

from app.account import routes
