import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TradeAnalyzer:
    @staticmethod
    def calculate_metrics(trades):
        if not trades:
            # Return default metrics for empty trade history
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'profit_factor': 0.0,
                'avg_winning_trade': 0.0,
                'avg_losing_trade': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
            
        # Filter out trades without P&L data
        valid_trades = [t for t in trades if t.pnl is not None]
        
        if not valid_trades:
            return {
                'total_trades': len(trades),
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'profit_factor': 0.0,
                'avg_winning_trade': 0.0,
                'avg_losing_trade': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
        
        df = pd.DataFrame([{
            'pnl': trade.pnl or 0,
            'pnl_percent': trade.pnl_percent or 0,
            'entry_time': trade.entry_time,
            'exit_time': trade.exit_time,
            'size': trade.size,
            'symbol': trade.symbol
        } for trade in valid_trades])
        
        # Calculate basic metrics
        winning_trades = len([t for t in valid_trades if t.pnl > 0])
        losing_trades = len([t for t in valid_trades if t.pnl < 0])
        total_trades = len(valid_trades)
        
        # Calculate win rate
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate P&L metrics
        total_pnl = df['pnl'].sum()
        avg_pnl = df['pnl'].mean()
        largest_win = df['pnl'].max()
        largest_loss = df['pnl'].min()
        
        # Calculate profit factor
        wins = df[df['pnl'] > 0]['pnl'].sum()
        losses = abs(df[df['pnl'] < 0]['pnl'].sum())
        profit_factor = wins / losses if losses > 0 else float('inf')
        
        # Calculate average winning and losing trades
        avg_winning_trade = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_losing_trade = df[df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Calculate max drawdown
        equity_curve = TradeAnalyzer.generate_equity_data(valid_trades)
        if equity_curve:
            equity_values = [point['equity'] for point in equity_curve]
            running_max = np.maximum.accumulate(equity_values)
            drawdowns = (equity_values - running_max) / running_max * 100
            max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0
        else:
            max_drawdown = 0
        
        # Calculate Sharpe ratio (simplified)
        if len(df) > 1:
            returns = df['pnl_percent'] / 100  # Convert percentage to decimal
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_pnl': round(avg_pnl, 2),
            'largest_win': round(largest_win, 2),
            'largest_loss': round(largest_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'avg_winning_trade': round(avg_winning_trade, 2),
            'avg_losing_trade': round(avg_losing_trade, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 2)
        }
        
        return metrics
    
    @staticmethod
    def generate_equity_data(trades):
        if not trades:
            return []
            
        equity_curve = []
        running_balance = 0
        
        # Sort trades by exit time (use entry time for open trades)
        sorted_trades = sorted(trades, 
                             key=lambda x: x.exit_time if x.exit_time else x.entry_time)
        
        for trade in sorted_trades:
            if trade.pnl is not None:
                running_balance += trade.pnl
                equity_curve.append({
                    'date': (trade.exit_time if trade.exit_time else trade.entry_time).isoformat(),
                    'equity': round(running_balance, 2)
                })
            
        return equity_curve
    
    @staticmethod
    def get_recent_trades(trades, limit=5):
        """Get most recent trades sorted by entry time"""
        if not trades:
            return []
        
        return sorted(trades, key=lambda x: x.entry_time, reverse=True)[:limit]
    
    @staticmethod
    def analyze_by_symbol(trades):
        """Analyze performance by symbol"""
        if not trades:
            return {}
            
        symbol_data = {}
        
        for trade in trades:
            if trade.pnl is None:
                continue
                
            symbol = trade.symbol
            if symbol not in symbol_data:
                symbol_data[symbol] = {
                    'trades': 0,
                    'winning_trades': 0,
                    'total_pnl': 0,
                    'pnl_values': []
                }
            
            symbol_data[symbol]['trades'] += 1
            symbol_data[symbol]['total_pnl'] += trade.pnl
            symbol_data[symbol]['pnl_values'].append(trade.pnl)
            
            if trade.pnl > 0:
                symbol_data[symbol]['winning_trades'] += 1
        
        # Calculate additional metrics for each symbol
        for symbol, data in symbol_data.items():
            data['win_rate'] = (data['winning_trades'] / data['trades'] * 100) if data['trades'] > 0 else 0
            data['avg_pnl'] = data['total_pnl'] / data['trades'] if data['trades'] > 0 else 0
            
        return symbol_data

    @staticmethod
    def calculate_account_metrics(trades, account_balances):
        """Calculate performance metrics including account balance"""
        trade_metrics = TradeAnalyzer.calculate_metrics(trades)
    
        if not account_balances:
            trade_metrics['current_balance'] = 0
            trade_metrics['total_deposits'] = 0
            trade_metrics['net_profit'] = trade_metrics['total_pnl']
            return trade_metrics
    
        current_balance = account_balances[-1].balance
        initial_balance = account_balances[0].balance
    
        # Calculate net profit (current balance minus initial deposits)
        # This is a simplified calculation - you might want to track deposits/withdrawals separately
        net_profit = current_balance - initial_balance + trade_metrics['total_pnl']
    
        trade_metrics.update({
            'current_balance': current_balance,
            'initial_balance': initial_balance,
            'net_profit': net_profit,
            'total_return': ((current_balance - initial_balance) / initial_balance * 100) if initial_balance > 0 else 0
        })
    
        return trade_metrics

    @staticmethod
    def analyze_by_timeframe(trades):
        """Analyze performance by different time periods"""
        if not trades:
            return {}

        time_data = {
        'by_hour': {},
        'by_day': {},
        'by_month': {}
        }

        for trade in trades:
            if trade.entry_time:
                # By hour
                hour = trade.entry_time.hour
                if hour not in time_data['by_hour']:
                    time_data['by_hour'][hour] = {'trades': 0, 'total_pnl': 0, 'winning': 0}
                    time_data['by_hour'][hour]['trades'] += 1
                    time_data['by_hour'][hour]['total_pnl'] += trade.pnl or 0
                    if trade.pnl and trade.pnl > 0:
                        time_data['by_hour'][hour]['winning'] += 1

                # By day of week
                day = trade.entry_time.strftime('%A')
                if day not in time_data['by_day']:
                    time_data['by_day'][day] = {'trades': 0, 'total_pnl': 0, 'winning': 0}
                    time_data['by_day'][day]['trades'] += 1
                    time_data['by_day'][day]['total_pnl'] += trade.pnl or 0
                    if trade.pnl and trade.pnl > 0:
                        time_data['by_day'][day]['winning'] += 1
        return time_data

    @staticmethod
    def calculate_risk_metrics(trades, account_balances):
        """Calculate risk management metrics"""
        if not trades:
            return {}

        # Average risk per trade
        risk_per_trade = []
        for trade in trades:
            if trade.pnl is not None:
                # Calculate risk as percentage of account (simplified)
                risk_per_trade.append(abs(trade.pnl) / 1000)  # Adjust based on your risk model

        # Win streak and loss streak
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        last_win = None

        for trade in sorted(trades, key=lambda x: x.entry_time):
            if trade.pnl is not None:
                is_win = trade.pnl > 0
                if last_win is None or is_win == last_win:
                    current_streak += 1
                else:
                    current_streak = 1

                if is_win:
                    max_win_streak = max(max_win_streak, current_streak)
                else:
                    max_loss_streak = max(max_loss_streak, current_streak)

                last_win = is_win

        return {
        'avg_risk_per_trade': sum(risk_per_trade) / len(risk_per_trade) if risk_per_trade else 0,
        'max_win_streak': max_win_streak,
        'max_loss_streak': max_loss_streak,
        'risk_reward_ratio': 0,  # Calculate based on your strategy
        'expectancy': 0  # (Win% * Avg Win) - (Loss% * Avg Loss)
    }
