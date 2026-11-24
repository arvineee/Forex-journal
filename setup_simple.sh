#!/bin/bash
echo "🚀 Setting up Forex Journal on Termux (Simplified)..."

# Install Python if not already installed
pkg install python -y

# Upgrade pip
pip install --upgrade pip

# Install required packages one by one to avoid conflicts
echo "📦 Installing Python packages..."
pip install Flask==2.3.3
pip install Flask-SQLAlchemy==3.0.5
pip install Flask-Login==0.6.3
pip install Flask-WTF==1.2.1
pip install Flask-Migrate==4.0.4
pip install Werkzeug==2.3.7
pip install python-dateutil==2.8.2
pip install python-dotenv==1.0.0

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p app/static/uploads
mkdir -p migrations

# Create .env file
echo "🔧 Creating environment configuration..."
cat > .env << EOL
SECRET_KEY=your-super-secret-key-change-this-in-production
DATABASE_URL=sqlite:///forex_journal.db
EOL

# Initialize the database
echo "🗃️ Initializing database..."
python init_db_simple.py

echo "✅ Setup complete!"
echo "🎯 To run the application: python run.py"
echo "🌐 Then open: http://localhost:5000 in your browser"
