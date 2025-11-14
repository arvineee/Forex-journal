#!/usr/bin/env python3
"""
Management script for Forex Journal
Use this for common administrative tasks
"""
import os
import click
from flask_migrate import Migrate
from app import create_app, db
from app.models import User, Trade, TradeImage

app = create_app()
migrate = Migrate(app, db)

@app.shell_context_processor
def make_shell_context():
    return dict(db=db, User=User, Trade=Trade, TradeImage=TradeImage)

@app.cli.command()
def initdb():
    """Initialize the database"""
    db.create_all()
    click.echo('✅ Database initialized!')

@app.cli.command()
def dropdb():
    """Drop all database tables"""
    if click.confirm('⚠️  Are you sure you want to drop all tables? This will delete all data!'):
        db.drop_all()
        click.echo('✅ All tables dropped!')

@app.cli.command()
def create_admin():
    """Create an admin user"""
    username = click.prompt('Username')
    email = click.prompt('Email')
    password = click.prompt('Password', hide_input=True, confirmation_prompt=True)
    
    user = User(username=username, email=email)
    user.set_password(password)
    
    db.session.add(user)
    db.session.commit()
    click.echo(f'✅ Admin user {username} created!')

@app.cli.command()
def stats():
    """Show database statistics"""
    users = User.query.count()
    trades = Trade.query.count()
    images = TradeImage.query.count()
    
    click.echo(f'📊 Database Statistics:')
    click.echo(f'   Users: {users}')
    click.echo(f'   Trades: {trades}')
    click.echo(f'   Images: {images}')

if __name__ == '__main__':
    app.cli()
