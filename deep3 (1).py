# -*- coding: utf-8 -*-
import MetaTrader5 as mt5
import pandas as pd
import time
import logging
import os
from datetime import datetime

# --- Configuration ---
SYMBOLS = [
    {"symbol": "BTCUSD", "timeframe": mt5.TIMEFRAME_M1, "lot_size": 0.01, "slippage": 3},
    {"symbol": "XAUUSD", "timeframe": mt5.TIMEFRAME_M1, "lot_size": 0.01, "slippage": 5},
    {"symbol": "US500.cash", "timeframe": mt5.TIMEFRAME_M1, "lot_size": 0.5, "slippage": 2}
]

MAGIC_NUMBER = 12345
MAX_POSITIONS_PER_SYMBOL = 3

# Strategy Parameters
SMA_PERIOD = 20
LOOKAHEAD_PERIOD = 5
TP_RR_RATIO = 1.5
SL_BUFFER_POINTS = 0

# Bot Control
CHECK_INTERVAL_SECONDS = 30
MAX_RETRY_CONNECT = 5
RETRY_DELAY_SECONDS = 10
DATA_FETCH_COUNT = SMA_PERIOD + LOOKAHEAD_PERIOD * 2 + 50

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('trading_bot.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# --- Global State Variables ---
last_processed_t1_time = {symbol["symbol"]: None for symbol in SYMBOLS}
last_candle_time = {symbol["symbol"]: None for symbol in SYMBOLS}

def initialize_mt5():
    """Initialize MT5 connection with retry logic"""
    retry_count = 0
    while retry_count < MAX_RETRY_CONNECT:
        if mt5.initialize():
            logger.info("MT5 initialized successfully")
            return True
        logger.error(f"MT5 initialization failed, attempt {retry_count+1}/{MAX_RETRY_CONNECT}")
        time.sleep(RETRY_DELAY_SECONDS)
        retry_count += 1
    logger.critical("Failed to initialize MT5 after maximum retries")
    return False

def get_ohlc_data(symbol, timeframe, count):
    """Fetch OHLC data from MT5"""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        logger.warning(f"No data received for {symbol} {timeframe}")
        return None
    
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "tick_volume": "Volume"
    }, inplace=True)
    return df

def calculate_indicators(df):
    """Calculate technical indicators"""
    if df is None or "Volume" not in df.columns:
        return None
    df["Volume_SMA"] = df["Volume"].rolling(window=SMA_PERIOD).mean()
    return df

def validate_trade(symbol, price, sl_price, tp_price, lot_size):
    """Validate trade conditions"""
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        logger.error(f"Could not get symbol info for {symbol}")
        return False
    
    # Price validation
    if price <= 0 or sl_price <= 0 or tp_price <= 0:
        logger.warning(f"Invalid prices for {symbol}")
        return False
    
    # Lot size validation
    if lot_size < symbol_info.volume_min or lot_size > symbol_info.volume_max:
        logger.warning(f"Invalid lot size for {symbol}")
        return False
    
    # Stop distance validation
    min_stop_distance = 10 * symbol_info.point
    if abs(price - sl_price) < min_stop_distance:
        logger.warning(f"Stop loss too close for {symbol}")
        return False
    
    # Market status check
    if symbol_info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
        logger.warning(f"Market not open for {symbol}")
        return False
    
    return True

def safe_order_send(symbol, order_type, lot_size, sl_price, tp_price, slippage, magic=0, max_retries=3):
    """Safe order execution with retries"""
    retry_count = 0
    while retry_count < max_retries:
        try:
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                logger.error(f"Failed to get tick data for {symbol}")
                return None
            
            price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
            
            if not validate_trade(symbol, price, sl_price, tp_price, lot_size):
                return None
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": order_type,
                "price": price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": slippage,
                "magic": magic,
                "comment": "Python Trading Bot",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Order executed for {symbol}. Ticket: {result.order}")
                return result
            
            logger.warning(f"Order failed for {symbol}. Retry {retry_count+1}/{max_retries}")
            
        except Exception as e:
            logger.error(f"Error in order execution: {str(e)}")
        
        retry_count += 1
        time.sleep(1)
    
    logger.error(f"Failed to execute order for {symbol} after {max_retries} attempts")
    return None

def check_open_positions(symbol, magic):
    """Check existing positions for a symbol"""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return 0
    return len([pos for pos in positions if pos.magic == magic])

def check_patterns_and_trade(df, symbol, lot_size, slippage):
    """Enhanced pattern detection with strict entry conditions"""
    global last_processed_t1_time

    if df is None or len(df) < SMA_PERIOD + LOOKAHEAD_PERIOD + 2:
        return

    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        return
    
    point = symbol_info.point

    for i in range(len(df)-2, max(len(df)-(LOOKAHEAD_PERIOD+5), SMA_PERIOD-1), -1):
        b1_candle = df.iloc[i]
        
        # Basic validation
        if pd.isna(b1_candle["Volume"]) or pd.isna(b1_candle["Volume_SMA"]):
            continue
            
        # Volume filter
        if b1_candle["Volume"] < b1_candle["Volume_SMA"]:
            continue

        # Search for confirmation candle
        for j in range(1, LOOKAHEAD_PERIOD+1):
            t1_idx = i + j
            if t1_idx >= len(df):
                break
                
            t1 = df.iloc[t1_idx]
            
            # Condition 1: Close above b1's high
            if t1["Close"] > b1_candle["High"]:
                # Condition 2: Volume less than b1's volume
                if t1["Volume"] < b1_candle["Volume"]:
                    # Condition 3: No candle touched b1's low
                    valid_entry = True
                    for k in range(i+1, t1_idx):
                        if df.iloc[k]["Low"] <= b1_candle["Low"]:
                            valid_entry = False
                            break
                    
                    if valid_entry:
                        # Avoid duplicate signals
                        if last_processed_t1_time.get(symbol) and t1.name <= last_processed_t1_time[symbol]:
                            continue
                            
                        # Check position limit
                        if check_open_positions(symbol, MAGIC_NUMBER) >= MAX_POSITIONS_PER_SYMBOL:
                            logger.warning(f"Max positions reached for {symbol}")
                            continue

                        # Calculate trade levels
                        entry_price = (b1_candle["Low"] + t1["High"]) / 2
                        sl_price = b1_candle["Low"] - (SL_BUFFER_POINTS * point)
                        tp_price = entry_price + ((entry_price - sl_price) * TP_RR_RATIO)
                        
                        # Execute buy order
                        result = safe_order_send(
                            symbol=symbol,
                            order_type=mt5.ORDER_TYPE_BUY,
                            lot_size=lot_size,
                            sl_price=sl_price,
                            tp_price=tp_price,
                            slippage=slippage,
                            magic=MAGIC_NUMBER
                        )
                        
                        if result:
                            last_processed_t1_time[symbol] = t1.name
                            logger.info(f"New buy position opened for {symbol} at {entry_price}")
                        break
                break  # Exit after first candle closing above b1's high

def main():
    """Main trading loop"""
    try:
        if not initialize_mt5():
            return

        logger.info("Starting trading bot with enhanced entry conditions...")
        
        while True:
            for symbol_info in SYMBOLS:
                symbol = symbol_info["symbol"]
                
                if not mt5.terminal_info():
                    if not initialize_mt5():
                        break
                
                df = get_ohlc_data(symbol, symbol_info["timeframe"], DATA_FETCH_COUNT)
                
                if df is not None and not df.empty:
                    current_time = df.index[-1]
                    
                    if last_candle_time[symbol] is None or current_time > last_candle_time[symbol]:
                        df = calculate_indicators(df)
                        if df is not None:
                            check_patterns_and_trade(
                                df, 
                                symbol, 
                                symbol_info["lot_size"],
                                symbol_info["slippage"]
                            )
                        last_candle_time[symbol] = current_time

            time.sleep(CHECK_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.exception(f"Critical error: {str(e)}")
    finally:
        mt5.shutdown()
        logger.info("Bot shutdown complete")

if __name__ == "__main__":
    main()