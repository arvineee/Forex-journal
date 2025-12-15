import os
import uuid
from werkzeug.utils import secure_filename
from PIL import Image
from flask import current_app
from decimal import Decimal

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_image(file):
    if file and allowed_file(file.filename):
        # Generate unique filename
        file_ext = os.path.splitext(file.filename)[1]
        filename = str(uuid.uuid4()) + file_ext
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Open and optimize image
        img = Image.open(file)
        
        # Resize if too large (max 1200px width)
        if img.width > 1200:
            ratio = 1200 / img.width
            new_height = int(img.height * ratio)
            img = img.resize((1200, new_height), Image.Resampling.LANCZOS)
        
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        # Save optimized image
        img.save(file_path, 'JPEG', quality=85, optimize=True)
        
        return filename
    return None

def calculate_pnl(direction, entry_price, exit_price, size, symbol):
    """
    Calculate P&L with broker-accurate pip calculations.
    Based on actual broker trade data analysis.
    
    Args:
        direction: 'BUY' or 'SELL'
        entry_price: Entry price as float
        exit_price: Exit price as float or None
        size: Position size in lots (e.g., 0.01 = micro lot, 1.0 = standard lot)
        symbol: Trading symbol (e.g., 'EURUSD', 'XAUUSD', 'GBPJPY')
    
    Returns:
        tuple: (pnl, pnl_percent) or (None, None) if exit_price is None
    """
    if not exit_price:
        return None, None
    
    # Convert Decimal to float for calculations
    entry_price = float(entry_price) if isinstance(entry_price, Decimal) else entry_price
    exit_price = float(exit_price) if isinstance(exit_price, Decimal) else exit_price
    size = float(size) if isinstance(size, Decimal) else size
    
    symbol = symbol.upper().replace('/', '')
    
    # Calculate price difference based on direction
    if direction == 'BUY':
        price_diff = exit_price - entry_price
    else:  # SELL
        price_diff = entry_price - exit_price
    
    # ========== GOLD (XAU/USD) ==========
    if 'XAU' in symbol or 'GOLD' in symbol:
        # Gold: 1 pip = 0.01 movement
        # Based on broker data, for 0.01 lot:
        # 1 pip movement ≈ $0.01 per 0.01 lot
        # Standard lot (1.0) = 100 ounces
        pip_size = 0.01
        pips_moved = price_diff / pip_size
        
        # Pip value per standard lot = $1.00
        # For micro lot (0.01): pip value = $0.01
        pip_value_per_lot = 1.0
        pnl = pips_moved * pip_value_per_lot * size
        
    # ========== SILVER (XAG/USD) ==========
    elif 'XAG' in symbol or 'SILVER' in symbol:
        # From broker data: 0.01 lot, 49.180 to 49.140 (40 pips) = -$2.00
        # 40 pips × 0.01 lot = -$2.00
        # Therefore: 1 pip × 0.01 lot = $0.05
        # Standard lot pip value = $5.00 per pip
        pip_size = 0.001
        pips_moved = price_diff / pip_size
        
        # Pip value: $5.00 per standard lot
        pip_value_per_lot = 5.0
        pnl = pips_moved * pip_value_per_lot * size
    
    # ========== JPY PAIRS ==========
    elif 'JPY' in symbol:
        # From broker data: GBPJPY 0.01 lot, 201.500 to 201.451 (49 pips) = -$0.32
        # 49 pips × 0.01 lot = -$0.32
        # 1 pip × 0.01 lot = -$0.0065
        # Standard lot pip value ≈ $0.65 per pip
        pip_size = 0.01
        pips_moved = price_diff / pip_size
        
        # Broker's pip value for JPY pairs (varies by current USD/JPY rate)
        # Average observed: $0.65-0.70 per pip per standard lot
        pip_value_per_lot = 0.653  # Calibrated from GBPJPY data: 0.32 / 49 / 0.01
        pnl = pips_moved * pip_value_per_lot * size
    
    # ========== STANDARD FOREX PAIRS ==========
    else:
        # From broker data: EURCAD analysis
        # Trade 1: 0.01 lot, SELL 1.62834→1.63051 (21.7 pips) = -$1.53
        # Trade 2: 0.01 lot, BUY 1.62666→1.62136 (53.0 pips) = -$3.78
        # Trade 3: 0.01 lot, BUY 1.62603→1.62136 (46.7 pips) = -$3.33
        # Trade 4: 0.01 lot, BUY 1.62596→1.62136 (46.0 pips) = -$3.28
        
        # Average pip value calculation:
        # Trade 2: 3.78 / 53.0 / 0.01 = $0.713 per pip per lot
        # Trade 3: 3.33 / 46.7 / 0.01 = $0.713 per pip per lot
        # Trade 4: 3.28 / 46.0 / 0.01 = $0.713 per pip per lot
        
        pip_size = 0.0001
        pips_moved = price_diff / pip_size
        
        # Determine pip value based on quote currency
        if 'USD' in symbol:
            if symbol.endswith('USD'):
                # Quote currency is USD (e.g., EUR/USD, GBP/USD)
                # Standard lot = $10 per pip
                pip_value_per_lot = 10.0
            else:
                # Base currency is USD (e.g., USD/CAD, USD/CHF)
                # Need to convert, approximate based on rate
                # For USD/CAD at ~1.35: pip value ≈ $7.40 per lot
                pip_value_per_lot = 7.4
        elif 'CAD' in symbol:
            # Cross pairs with CAD (e.g., EUR/CAD, GBP/CAD)
            # From broker data: ≈ $0.713 per pip per 0.01 lot
            # Standard lot: $71.3 per pip
            pip_value_per_lot = 7.13
        elif 'EUR' in symbol or 'GBP' in symbol:
            # Cross pairs
            # Approximate based on major currency pairs
            pip_value_per_lot = 10.0
        else:
            # Default for other pairs
            pip_value_per_lot = 10.0
        
        pnl = pips_moved * pip_value_per_lot * size
    
    # ========== CALCULATE PERCENTAGE RETURN ==========
    # Position value = Entry price × Contract size × Lot size
    if 'XAU' in symbol or 'GOLD' in symbol:
        position_value = entry_price * 100 * size
    elif 'XAG' in symbol or 'SILVER' in symbol:
        position_value = entry_price * 5000 * size
    else:
        position_value = entry_price * 100000 * size
    
    # Calculate percentage
    pnl_percent = (pnl / position_value) * 100 if position_value > 0 else 0
    
    return round(pnl, 2), round(pnl_percent, 2)


def validate_trade_data(form_data):
    """
    Validate trade data before saving.
    
    Args:
        form_data: Dictionary containing trade form data
    
    Returns:
        tuple: (is_valid, error_message)
    """
    errors = []
    
    # Validate required fields
    if not form_data.get('symbol'):
        errors.append("Symbol is required")
    
    if not form_data.get('direction') or form_data['direction'] not in ['BUY', 'SELL']:
        errors.append("Valid direction (BUY/SELL) is required")
    
    if not form_data.get('entry_price') or float(form_data['entry_price']) <= 0:
        errors.append("Valid entry price is required")
    
    if not form_data.get('size') or float(form_data['size']) <= 0:
        errors.append("Valid position size is required")
    
    # Validate exit price if provided
    if form_data.get('exit_price'):
        try:
            exit_price = float(form_data['exit_price'])
            if exit_price <= 0:
                errors.append("Exit price must be greater than 0")
        except (ValueError, TypeError):
            errors.append("Invalid exit price format")
    
    # Validate dates
    if form_data.get('exit_time') and form_data.get('entry_time'):
        try:
            from datetime import datetime
            entry_time = datetime.fromisoformat(form_data['entry_time'])
            exit_time = datetime.fromisoformat(form_data['exit_time'])
            
            if exit_time < entry_time:
                errors.append("Exit time cannot be before entry time")
        except (ValueError, TypeError):
            errors.append("Invalid date format")
    
    if errors:
        return False, "; ".join(errors)
    
    return True, None


def format_currency(amount, decimals=2):
    """
    Format currency amount with proper decimal places.
    
    Args:
        amount: Numeric amount to format
        decimals: Number of decimal places (default 2)
    
    Returns:
        str: Formatted currency string
    """
    if amount is None:
        return "-"
    
    try:
        amount = float(amount)
        return f"${amount:,.{decimals}f}"
    except (ValueError, TypeError):
        return "-"


def format_percentage(value, decimals=2):
    """
    Format percentage value.
    
    Args:
        value: Numeric percentage value
        decimals: Number of decimal places (default 2)
    
    Returns:
        str: Formatted percentage string
    """
    if value is None:
        return "-"
    
    try:
        value = float(value)
        return f"{value:.{decimals}f}%"
    except (ValueError, TypeError):
        return "-"
