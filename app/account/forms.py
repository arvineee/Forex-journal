
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, MultipleFileField
from wtforms import StringField, FloatField, DateTimeField, TextAreaField, SelectField, IntegerField, SubmitField
from wtforms.validators import DataRequired, Optional, NumberRange
from datetime import datetime

class AccountBalanceForm(FlaskForm):
    balance = FloatField('Account Balance*', validators=[DataRequired()])
    date = DateTimeField('Date*', default=datetime.utcnow, 
                        validators=[DataRequired()], format='%Y-%m-%dT%H:%M')
    notes = TextAreaField('Notes', validators=[Optional()],
                         render_kw={"placeholder": "e.g., Initial deposit, Withdrawal, Monthly update"})
    submit = SubmitField('Save Balance')
