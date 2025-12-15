from flask_wtf import FlaskForm
from flask_wtf.file import FileField, MultipleFileField
from wtforms import StringField, FloatField, DateTimeField, TextAreaField, SelectField, IntegerField, SubmitField,DateField,BooleanField
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


class ExportForm(FlaskForm):
    format = SelectField('Format', choices=[
        ('csv', 'CSV (Excel Compatible)'),
        ('excel', 'Microsoft Excel (.xlsx)'),
        ('json', 'JSON (Structured Data)'),
        ('pdf', 'PDF Report')
    ], validators=[DataRequired()])
    
    date_range = SelectField('Time Period', choices=[
        ('all', 'All Trades'),
        ('today', 'Today'),
        ('week', 'Last 7 Days'),
        ('month', 'Last 30 Days'),
        ('quarter', 'Last 90 Days'),
        ('year', 'Last 365 Days'),
        ('custom', 'Custom Range')
    ], validators=[DataRequired()])
    
    start_date = DateField('Start Date', validators=[Optional()])
    end_date = DateField('End Date', validators=[Optional()])
    
    include_images = BooleanField('Include Image Information')
    include_annotations = BooleanField('Include Annotation Data')
    include_summary = BooleanField('Include Summary Statistics')
    
    submit = SubmitField('Generate Export')
