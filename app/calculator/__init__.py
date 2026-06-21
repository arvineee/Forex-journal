from flask import Blueprint

calculator_bp = Blueprint(
    'calculator',
    __name__,
    template_folder='templates'
)

from app.calculator import routes  # noqa: E402,F401

