

import numpy as np

def getMyPosition(prcSoFar, curPos, entry_prices):
    # RSI AND MACD MIXED 
    # --- Enhanced Parameters ---
    RSI_PERIOD = 25
    RSI_OVERSOLD = 15 
    RSI_OVERBOUGHT = 71  
    MACD_FAST = 13  
    MACD_SLOW = 26  
    MACD_SIGNAL = 9
    POSITION_SIZE = 10000
    STOP_LOSS_PCT = 0.01  
    TAKE_PROFIT_PCT = 0.01
    VOLATILITY_LOOKBACK = 20  

    # --- Initialization ---
    num_instruments, num_days = prcSoFar.shape
    
    new_positions = np.copy(curPos)

    min_required = max(RSI_PERIOD, MACD_SLOW) + MACD_SIGNAL + 1
    if num_days < min_required:
        return new_positions

    # --- Main Loop for Each Instrument ---
    for i in range(num_instruments):
        # Calculate indicators once per instrument
        prices = prcSoFar[i, :]
        current_price = prices[-1]
        rsi_values = calculate_rsi(prices, RSI_PERIOD)
        macd_line, signal_line = calculate_macd(prices, MACD_FAST, MACD_SLOW, MACD_SIGNAL)

        
        if len(rsi_values) < 2 or len(macd_line) < 2:
            continue

        # Calculate volatility for position sizing
        volatility = calculate_volatility(prices, VOLATILITY_LOOKBACK)

        # --- PART 1: MANAGE EXISTING POSITIONS ---
        if curPos[i] != 0:
            # Check for a price-based exit (Stop-Loss or Take-Profit)
            exit_triggered = check_exit_conditions(
                entry_prices[i], curPos[i], current_price, STOP_LOSS_PCT, TAKE_PROFIT_PCT
            )
            if exit_triggered:
                new_positions[i] = 0  # Signal to close

        # --- PART 2: LOOK FOR NEW ENTRY SIGNALS ---
        else: 
           
            if not is_trending_market(prices):
                continue 
            
            
            signal = generate_signal(
                rsi_values[-1], rsi_values[-2],
                macd_line[-1], signal_line[-1],
                macd_line[-2], signal_line[-2],
                (macd_line[-1] - signal_line[-1]), (macd_line[-2] - signal_line[-2]),
                RSI_OVERSOLD, RSI_OVERBOUGHT
            )

            
            if signal != 0:
                strength = calculate_signal_strength(
                    rsi_values[-1], macd_line[-1], signal_line[-1], current_price
                )
                new_positions[i] = calculate_position_size(
                    signal, POSITION_SIZE, current_price, strength, volatility
                )

    return new_positions


def check_exit_conditions(entry_price, direction, current_price, stop_loss_pct, take_profit_pct):
    
    # --- Check for a BUY position ---
    if direction > 0: # Buy position
        stop_loss_price = entry_price * (1 - stop_loss_pct)
        take_profit_price = entry_price * (1 + take_profit_pct)
        if current_price <= stop_loss_price or current_price >= take_profit_price:
            return True

    # --- Check for a SELL position ---
    elif direction < 0: # Sell position
        stop_loss_price = entry_price * (1 + stop_loss_pct)
        take_profit_price = entry_price * (1 - take_profit_pct)
        if current_price >= stop_loss_price or current_price <= take_profit_price:
            return True

    return False # No exit condition met


def is_trending_market(prices, lookback=20):
 
    if len(prices) < lookback + 1:
        return True  # Default to trending if not enough data
    
    # Calculate returns over lookback period
    returns = np.diff(prices[-lookback:])
    
    # Check if average return magnitude is greater than volatility
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    
    # Market is trending if mean return is significant vs volatility
    return abs(avg_return) > std_return * 0.5


def calculate_volatility(prices, lookback=20):
    
    if len(prices) < lookback + 1:
        return 1.0  # Default volatility
    
    # Calculate returns and their standard deviation
    returns = np.diff(prices[-lookback:])
    volatility = np.std(returns)
    
    
    # Higher volatility = higher multiplier
    return max(0.5, min(2.0, volatility / 0.02))  # Cap between 0.5x and 2.0x


# HELPER FUNCTIONS

def calculate_rsi(prices, period=14):
    if len(prices) < period + 1: return np.array([])
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gains = np.zeros(len(gains)); avg_losses = np.zeros(len(losses))
    avg_gains[period-1] = np.mean(gains[:period]); avg_losses[period-1] = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_gains[i] = (avg_gains[i-1] * (period-1) + gains[i]) / period
        avg_losses[i] = (avg_losses[i-1] * (period-1) + losses[i]) / period
    rs = np.divide(avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses!=0)
    rsi = 100 - (100 / (1 + rs))
    return rsi[period-1:]


def calculate_ema(data, period):
    if len(data) < period: return np.array([])
    multiplier = 2 / (period + 1)
    ema = np.zeros(len(data))
    ema[period-1] = np.mean(data[:period])
    for i in range(period, len(data)):
        ema[i] = (data[i] * multiplier) + (ema[i-1] * (1 - multiplier))
    return ema[period-1:]


def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    if len(prices) < slow_period + signal_period: return np.array([]), np.array([])
    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)
    min_length = min(len(ema_fast), len(ema_slow))
    ema_fast = ema_fast[-min_length:]; ema_slow = ema_slow[-min_length:]
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_period)
    if len(signal_line) < len(macd_line): macd_line = macd_line[-len(signal_line):]
    return macd_line, signal_line


def generate_signal(rsi_curr, rsi_prev, macd_curr, macd_signal_curr,
                   macd_prev, macd_signal_prev, hist_curr, hist_prev,
                   rsi_oversold=30, rsi_overbought=70):
    rsi_buy = rsi_curr <= rsi_oversold or (rsi_prev <= rsi_oversold and rsi_curr > rsi_oversold)
    macd_buy = macd_curr > macd_signal_curr and ((macd_prev <= macd_signal_prev and macd_curr > macd_signal_curr) or hist_curr > 0)
    if rsi_buy and macd_buy: 
        return 1
    rsi_sell = rsi_curr >= rsi_overbought or (rsi_prev >= rsi_overbought and rsi_curr < rsi_overbought)
    macd_sell = macd_curr < macd_signal_curr and ((macd_prev >= macd_signal_prev and macd_curr < macd_signal_curr) or hist_curr < 0)
    if rsi_sell and macd_sell: 
        return -1
    return 0


def calculate_signal_strength(rsi, macd, macd_signal, price):
    rsi_strength = abs(rsi - 50) / 50
    macd_divergence = abs(macd - macd_signal)
    macd_strength = min(macd_divergence / (price * 0.01), 1)
    combined_strength = (rsi_strength * 0.4) + (macd_strength * 0.6)
    return combined_strength


def calculate_position_size(signal, base_size, current_price, strength=1.0, volatility=1.0):
    """
    Enhanced position sizing with volatility adjustment.
    """
    # Reduce size during high volatility periods
    vol_adjustment = min(1.0, 1.0 / volatility)
    adjusted_size = base_size * (0.5 + 0.5 * strength) * vol_adjustment
    position = signal * adjusted_size
    max_units = int(10000 / (current_price + 1e-6))
    if position > 0:
        return min(position, max_units)
    else:
        return max(position, -max_units)


if __name__ == "__main__":
    np.random.seed(42)
    num_instruments = 50
    num_days = 200

    # Generate sample price data
    sample_prices = np.zeros((num_instruments, num_days))
    for i in range(num_instruments):
        base_price = 50 + np.random.rand() * 100
        returns = np.random.randn(num_days) * 0.02
        trend = np.random.randn() * 0.001
        sample_prices[i, :] = base_price * np.exp(np.cumsum(returns + trend))

    # --- Backtesting Loop ---
    print("Running enhanced backtest...")
    open_positions = {} # Key: instrument index, Value: {'direction': 1/-1, 'entry_price': float}
    min_hist = max(27, 26) + 9 + 1 # Minimum history needed for indicators

    for day in range(min_hist, num_days):
        # Provide the strategy with price history up to the current day
        price_history_slice = sample_prices[:, :day+1]

        # Get actions from the strategy function
        actions = getMyPosition(price_history_slice, open_positions)

        # Execute actions
        for action in actions:
            instrument_idx = action['instrument']
            if action['action'] == 'OPEN':
                if instrument_idx not in open_positions: # Ensure we don't open an existing position
                    open_positions[instrument_idx] = {
                        'direction': action['direction'],
                        'entry_price': sample_prices[instrument_idx, day]
                    }
                    print(f"Day {day}: OPEN {'BUY' if action['direction'] == 1 else 'SELL'} on Instrument {instrument_idx} at {open_positions[instrument_idx]['entry_price']:.2f}")

            elif action['action'] == 'CLOSE':
                if instrument_idx in open_positions: # Ensure we close an existing position
                    entry_price = open_positions[instrument_idx]['entry_price']
                    exit_price = sample_prices[instrument_idx, day]
                    profit = (exit_price - entry_price) * open_positions[instrument_idx]['direction']
                    print(f"Day {day}: CLOSE Instrument {instrument_idx} at {exit_price:.2f}. Profit/Loss: {profit:.2f}")
                    del open_positions[instrument_idx] # Remove from open positions

    print("\nEnhanced backtest finished.")
    print(f"Positions still open at the end: {len(open_positions)}")