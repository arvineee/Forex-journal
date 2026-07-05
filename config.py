import os
from datetime import timedelta

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') 
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(basedir, 'app/static/uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)
    
    # Migration directory
    MIGRATIONS_DIR = os.path.join(basedir, 'migrations')

    COUNTRIES_LIST = [
    {'code': 'KE', 'name': 'Kenya'},
    {'code': 'US', 'name': 'United States'},
    {'code': 'GB', 'name': 'United Kingdom'},
    {'code': 'CA', 'name': 'Canada'},
    {'code': 'AU', 'name': 'Australia'},
    {'code': 'IN', 'name': 'India'},
    {'code': 'NG', 'name': 'Nigeria'},
    {'code': 'ZA', 'name': 'South Africa'},
    {'code': 'UG', 'name': 'Uganda'},
    {'code': 'TZ', 'name': 'Tanzania'},
    {'code': 'DE', 'name': 'Germany'},
    {'code': 'FR', 'name': 'France'},
    {'code': 'NL', 'name': 'Netherlands'},
    {'code': 'AE', 'name': 'United Arab Emirates'},
    {'code': 'SG', 'name': 'Singapore'},
    {'code': 'JP', 'name': 'Japan'},
    {'code': 'CN', 'name': 'China'},
    {'code': 'RU', 'name': 'Russia'},
    {'code': 'BR', 'name': 'Brazil'},
    {'code': 'MX', 'name': 'Mexico'},
    {'code': 'PH', 'name': 'Philippines'},
    {'code': 'PK', 'name': 'Pakistan'},
    {'code': 'BD', 'name': 'Bangladesh'},
    {'code': 'ID', 'name': 'Indonesia'},
    {'code': 'VN', 'name': 'Vietnam'},
    {'code': 'EG', 'name': 'Egypt'},
    {'code': 'GH', 'name': 'Ghana'},
    {'code': 'ET', 'name': 'Ethiopia'},
    {'code': 'RW', 'name': 'Rwanda'},
    {'code': 'ZW', 'name': 'Zimbabwe'},
]
