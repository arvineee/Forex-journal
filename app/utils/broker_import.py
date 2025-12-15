# Create app/utils/broker_import.py
import csv
import json
from datetime import datetime

class BrokerImporter:
    @staticmethod
    def import_from_csv(file, user_id, broker_type='generic'):
        """Import trades from broker CSV export"""
        trades = []
        
        if broker_type == 'metatrader':
            trades = BrokerImporter._parse_metatrader_csv(file, user_id)
        elif broker_type == 'generic':
            trades = BrokerImporter._parse_generic_csv(file, user_id)
        
        return trades
    
    @staticmethod
    def _parse_metatrader_csv(file, user_id):
        trades = []
        reader = csv.DictReader(file)
        
        for row in reader:
            trade = {
                'symbol': row.get('Symbol', ''),
                'direction': 'BUY' if row.get('Type', '').lower() == 'buy' else 'SELL',
                'entry_price': float(row.get('Open Price', 0)),
                'exit_price': float(row.get('Close Price', 0)),
                'size': float(row.get('Volume', 0)),
                'entry_time': datetime.strptime(row.get('Open Time', ''), '%Y.%m.%d %H:%M:%S'),
                'exit_time': datetime.strptime(row.get('Close Time', ''), '%Y.%m.%d %H:%M:%S'),
                'pnl': float(row.get('Profit', 0)),
                'user_id': user_id
            }
            trades.append(trade)
        
        return trades
