from flask_wtf import FlaskForm
from flask_wtf.file import FileField, MultipleFileField
from wtforms import StringField, FloatField, DateTimeField, TextAreaField, SelectField, IntegerField, SubmitField
from wtforms.validators import DataRequired, Optional, NumberRange
from datetime import datetime

class TradeForm(FlaskForm):
    symbol = StringField('Currency Pair*', validators=[DataRequired()], 
                        render_kw={"placeholder": "e.g., EUR/USD, GBP/JPY, XAU/USD, XAG/USD"})
    direction = SelectField('Direction*', choices=[('BUY', 'BUY'), ('SELL', 'SELL')], 
                           validators=[DataRequired()])
    entry_price = FloatField('Entry Price*', validators=[DataRequired()])
    exit_price = FloatField('Exit Price', validators=[Optional()])
    size = FloatField('Position Size (Lots)*', validators=[DataRequired()])
    entry_time = DateTimeField('Entry Time*', default=datetime.utcnow, 
                              validators=[DataRequired()], format='%Y-%m-%dT%H:%M')
    exit_time = DateTimeField('Exit Time', validators=[Optional()], format='%Y-%m-%dT%H:%M')
    strategy = StringField('Trading Strategy', validators=[Optional()],
                          render_kw={"placeholder": "e.g., Breakout, Scalping"})
    timeframe = SelectField('Timeframe', choices=[
        ('', 'Select Timeframe'),
        ('1m', '1 Minute'),
        ('5m', '5 Minutes'),
        ('15m', '15 Minutes'),
        ('30m', '30 Minutes'),
        ('1h', '1 Hour'),
        ('4h', '4 Hours'),
        ('1d', '1 Day'),
        ('1w', '1 Week')
    ], validators=[Optional()])
    notes = TextAreaField('Trade Notes', validators=[Optional()],
                         render_kw={"placeholder": "What was your reasoning? Market conditions?"})
    emotions = SelectField('Emotional State', choices=[
        ('', 'Select Emotion'),
        ('confident', '😊 Confident'),
        ('neutral', '😐 Neutral'),
        ('anxious', '😰 Anxious'),
        ('greedy', '🤢 Greedy'),
        ('fearful', '😨 Fearful'),
        ('disciplined', '🎯 Disciplined'),
        ('impulsive', '⚡ Impulsive')
    ], validators=[Optional()])
    mistakes = TextAreaField('Mistakes & Learnings', validators=[Optional()],
                           render_kw={"placeholder": "What could you have done better?"})
    rating = SelectField('Self Rating', choices=[
        (1, '⭐ - Poor'),
        (2, '⭐⭐ - Below Average'),
        (3, '⭐⭐⭐ - Average'),
        (4, '⭐⭐⭐⭐ - Good'),
        (5, '⭐⭐⭐⭐⭐ - Excellent')
    ], coerce=int, validators=[Optional()])
    images = MultipleFileField('Chart Screenshots', validators=[Optional()])
    submit = SubmitField('Save Trade')
