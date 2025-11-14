#!/usr/bin/env python3
"""
Test script to verify the Forex Journal app works correctly
"""
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing imports...")
    try:
        from app import create_app, db
        from app.models import User, Trade, TradeImage
        from app.utils.analysis import TradeAnalyzer
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_database():
    """Test database connection and models"""
    print("🔍 Testing database...")
    try:
        from app import create_app, db
        app = create_app()
        
        with app.app_context():
            # Test database connection
            db.create_all()
            print("✅ Database connection successful!")
            
            # Test models
            user_count = User.query.count()
            trade_count = Trade.query.count()
            print(f"✅ Models working - Users: {user_count}, Trades: {trade_count}")
            
        return True
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_trade_analyzer():
    """Test the TradeAnalyzer utility"""
    print("🔍 Testing TradeAnalyzer...")
    try:
        from app.utils.analysis import TradeAnalyzer
        
        # Test with empty data
        metrics = TradeAnalyzer.calculate_metrics([])
        assert 'win_rate' in metrics
        assert 'total_trades' in metrics
        print("✅ TradeAnalyzer handles empty data correctly")
        
        # Test equity data generation
        equity_data = TradeAnalyzer.generate_equity_data([])
        assert isinstance(equity_data, list)
        print("✅ Equity data generation works")
        
        return True
    except Exception as e:
        print(f"❌ TradeAnalyzer error: {e}")
        return False

if __name__ == '__main__':
    print("🧪 Running Forex Journal tests...")
    print("-" * 50)
    
    tests_passed = 0
    tests_total = 3
    
    if test_imports():
        tests_passed += 1
    
    if test_database():
        tests_passed += 1
    
    if test_trade_analyzer():
        tests_passed += 1
    
    print("-" * 50)
    print(f"📊 Tests passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("🎉 All tests passed! The app should work correctly.")
        print("🚀 You can now run: python run.py")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
