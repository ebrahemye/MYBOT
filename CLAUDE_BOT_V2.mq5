//+------------------------------------------------------------------+
//|                                    XAUUSD Volume Breakout Bot   |
//|                                             Copyright 2025       |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025"
#property link      ""
#property version   "1.01"

#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\OrderInfo.mqh>

// Input parameters
input double LotSize = 0.01;
input string TradingSymbol = "XAUUSD";

// Trade objects
CTrade trade;
CSymbolInfo symbolInfo;
CPositionInfo positionInfo;
COrderInfo orderInfo;

// Global variables for H1 analysis
datetime lastH1Time = 0;
bool B1Found = false;
double B1High = 0;
double B1Low = 0;
long B1Volume = 0;
datetime B1Time = 0;


bool T1Found = false;
double T1High = 0;
double T1Low = 0;
double T1Close = 0;
long T1Volume = 0;
datetime T1Time = 0;
int T1Direction = 0; // 1 for buy, -1 for sell

// Add counter to track candles after B1
int candlesAfterB1 = 0;

// Fibonacci levels for H1
double FibLevel0 = 0;
double FibLevel1 = 0;
double CancelLevelMinus01 = 0;
double CancelLevel423 = 0;

// Global variables for M5 analysis
bool M5Active = false;
bool B1TouchedOnce = false; // NEW: Track if B1 was touched once
bool B11Found = false;
double B11High = 0;
double B11Low = 0;
long B11Volume = 0;
datetime B11Time = 0;

bool T11Found = false;
double T11High = 0;
double T11Low = 0;
double T11Close = 0;
long T11Volume = 0;
datetime T11Time = 0;

// M5 Fibonacci levels
double M5FibLevel0 = 0;
double M5FibLevel1 = 0;
double EntryLevel = 0;
double StopLoss = 0;
double TakeProfit = 0;

int candleCount = 0;
datetime lastM5Time = 0;

 void ResetB11()
{
    B11Found = false;
    B11High = 0;
    B11Low = 0;
    B11Volume = 0;
    B11Time = 0;
    candleCount = 0;
    T11Found = false;
}
void ResetB1()
{
    B1Found = false;
    B1High = 0;
    B1Low = 0;
    B1Volume = 0;
    B1Time = 0;
    T1Found = false;
    T1Direction = 0;
    M5Active = false;
    B1TouchedOnce = false; // Reset touch flag
    candlesAfterB1 = 0; // Reset counter
    ResetB11(); // Also reset M5 variables
    Print("تم إعادة تعيين B1 ومتغيرات M5");
}

void ResetAll()
{
    ResetB1();
    ResetB11();
    Print("تم إعادة تعيين جميع المتغيرات");
}

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    // Set symbol for trading
    symbolInfo.Name(TradingSymbol);
    trade.SetExpertMagicNumber(12345);
   
    Print("XAUUSD Volume Breakout Bot started");
    Print("Trading Symbol: ", TradingSymbol);
    Print("Lot Size: ", LotSize);
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("Bot stopped");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // Check for new H1 candle
    datetime currentH1Time = iTime(TradingSymbol, PERIOD_H1, 0);
    if(currentH1Time != lastH1Time)
    {
        lastH1Time = currentH1Time;
        AnalyzeH1();
    }
  
    // Check cancellation levels if B1 and T1 are found
    if(B1Found && T1Found)
    {
        CheckCancellationLevels();
    }
  
    // M5 analysis if activated
    if(M5Active)
    {
        // Check if B1 was touched (only once)
        if(!B1TouchedOnce)
        {
            if(CheckB1Touch())
            {
                B1TouchedOnce = true;
                string touchMessage = (T1Direction == 1) ? "تم لمس قمة B1 - بدء مراقبة M5" : "تم لمس قاع B1 - بدء مراقبة M5";
                Print(touchMessage);
            }
            else
            {
                return; // Wait for B1 touch
            }
        }
      
        // Check for new M5 candle
        datetime currentM5Time = iTime(TradingSymbol, PERIOD_M5, 0);
        if(currentM5Time != lastM5Time)
        {
            lastM5Time = currentM5Time;
            AnalyzeM5();
        }
    }
}

//+------------------------------------------------------------------+
//| Analyze H1 timeframe                                            |
//+------------------------------------------------------------------+
void AnalyzeH1()
{
    if(!B1Found)
    {
        SearchForB1();
    }
    else if(!T1Found)
    {
        // Increment counter when new H1 candle forms after B1
        candlesAfterB1++;
        SearchForT1();
    }
}

//+------------------------------------------------------------------+
//| Calculate average volume for given timeframe                    |
//+------------------------------------------------------------------+
double CalculateAverageVolume(ENUM_TIMEFRAMES timeframe, int period = 20)
{
    long totalVolume = 0;
    int validCandles = 0;
  
    for(int i = 1; i <= period + 5; i++) // +5 to ensure enough data
    {
        long vol = iVolume(TradingSymbol, timeframe, i);
        if(vol > 0) // Ignore candles without volume
        {
            totalVolume += vol;
            validCandles++;
          
            if(validCandles >= period) break;
        }
    }
  
    return (validCandles > 0) ? (double)totalVolume / validCandles : 0;
}

//+------------------------------------------------------------------+
//| Search for B1 candle on H1                                     |
//+------------------------------------------------------------------+
void SearchForB1()
{
    // Get last closed candle on H1
    long lastVolume = iVolume(TradingSymbol, PERIOD_H1, 1);
    datetime lastTime = iTime(TradingSymbol, PERIOD_H1, 1);
  
    if(lastVolume <= 0) return; // No data available
  
    // Calculate average volume of last 20 candles
    double avgVolume = CalculateAverageVolume(PERIOD_H1, 20);
  
    if(avgVolume <= 0) return; // Cannot calculate average
  
    // Check if last candle volume is >= average volume
    if((double)lastVolume >= avgVolume)
    {
        B1Found = true;
        B1High = iHigh(TradingSymbol, PERIOD_H1, 1);
        B1Low = iLow(TradingSymbol, PERIOD_H1, 1);
        B1Volume = lastVolume;
        B1Time = lastTime;
        candlesAfterB1 = 0; // Reset counter
      
        Print("تم ايجاد B1 حجمها ", (double)B1Volume, " متوسط الحجم ", avgVolume);
        Print("B1 High: ", B1High, " B1 Low: ", B1Low); // إضافة تفاصيل أكثر للديبج
    }
}

//+------------------------------------------------------------------+
//| Find candle index by time on H1                                |
//+------------------------------------------------------------------+
int FindH1CandleIndexByTime(datetime targetTime)
{
    int maxBars = iBars(TradingSymbol, PERIOD_H1);
  
    for(int i = 0; i < maxBars; i++)
    {
        datetime candleTime = iTime(TradingSymbol, PERIOD_H1, i);
        if(candleTime == targetTime)
        {
            return i;
        }
        if(candleTime < targetTime) // We've gone too far back
        {
            break;
        }
    }
    return -1; // Not found
}

//+------------------------------------------------------------------+
//| Search for T1 candle (breakout candle) - FIXED VERSION         |
//+------------------------------------------------------------------+
void SearchForT1()
{
    // Check if we've exceeded 5 candles after B1
    if(candlesAfterB1 > 5)
    {
        Print("لم يحدث اختراق خلال 5 شموع - إعادة تعيين B1");
        ResetB1();
        return;
    }
   
    // Check the current closed candle (which is candlesAfterB1 position after B1)
    int candleIndex = candlesAfterB1; // This represents the closed candle after B1
   
    double candleClose = iClose(TradingSymbol, PERIOD_H1, candleIndex);
    double candleHigh = iHigh(TradingSymbol, PERIOD_H1, candleIndex);
    double candleLow = iLow(TradingSymbol, PERIOD_H1, candleIndex);
    double candleOpen = iOpen(TradingSymbol, PERIOD_H1, candleIndex);
  
    // إضافة تفاصيل الشمعة للديبج
    Print("فحص الشمعة [", candlesAfterB1, ":5] - Open: ", candleOpen, " High: ", candleHigh, " Low: ", candleLow, " Close: ", candleClose);
    Print("B1 High: ", B1High, " B1 Low: ", B1Low);
   
    // Check for breakout
    if(candleClose > B1High) // Buy signal
    {
        Print("فحص الشمعه [", candlesAfterB1, ":5] تحقق شرط الاختراق للأعلى - Close: ", candleClose, " > B1 High: ", B1High);
        if(CheckT1Conditions(candleIndex, 1, candlesAfterB1))
        {
            T1Direction = 1;
            T1High = candleHigh;
            T1Low = candleLow;
            T1Close = candleClose;
            T1Volume = iVolume(TradingSymbol, PERIOD_H1, candleIndex);
            T1Time = iTime(TradingSymbol, PERIOD_H1, candleIndex);
            T1Found = true;
            SetupFibonacci();
            M5Active = true;
            B1TouchedOnce = false; // Reset touch flag
            Print("تم تفعيل تحليل M5 - إشارة شراء");
            return;
        }
    }
    else if(candleClose < B1Low) // Sell signal
    {
        Print("فحص الشمعه [", candlesAfterB1, ":5] تحقق شرط الاختراق للأسفل - Close: ", candleClose, " < B1 Low: ", B1Low);
        if(CheckT1Conditions(candleIndex, -1, candlesAfterB1))
        {
            T1Direction = -1;
            T1High = candleHigh;
            T1Low = candleLow;
            T1Close = candleClose;
            T1Volume = iVolume(TradingSymbol, PERIOD_H1, candleIndex);
            T1Time = iTime(TradingSymbol, PERIOD_H1, candleIndex);
            T1Found = true;
            SetupFibonacci();
            M5Active = true;
            B1TouchedOnce = false; // Reset touch flag
            Print("تم تفعيل تحليل M5 - إشارة بيع");
            return;
        }
    }
    else
    {
        Print("فحص الشمعه [", candlesAfterB1, ":5] لم يتحقق شرط الاختراق - Close في المنتصف");
        Print("Close: ", candleClose, " بين B1 High: ", B1High, " و B1 Low: ", B1Low);
    }
}

//+------------------------------------------------------------------+
//| Check T1 conditions (volume and internal touch)                |
//+------------------------------------------------------------------+
bool CheckT1Conditions(int candleIndex, int direction, int candleNum)
{
    string directionText = (direction == 1) ? "اعلى قمة B1" : "تحت قاع B1";
    Print("فحص الشمعه [", candleNum, ":5] تحقق شرط الاختراق ", directionText);
  
    long candleVolume = iVolume(TradingSymbol, PERIOD_H1, candleIndex);
  
    // Check volume condition - T1 volume should be LESS than B1 volume
    if(candleVolume >= B1Volume)
    {
        Print("فحص الشمعه[", candleNum, ":5] لم يتحقق الشرط الثاني - حجم الشمعه ", (double)candleVolume, " >= حجم شمعة B1 ", (double)B1Volume);
        // Reset B1 and start from this candle (it might be new B1)
        ResetB1();
        return false;
    }
  
    Print("فحص الشمعه[", candleNum, ":5] تحقق الشرط الثاني - حجم الشمعه ", (double)candleVolume, " < حجم شمعة B1 ", (double)B1Volume);
  
    // Check internal touch condition
    if(!CheckInternalTouchH1(candleIndex, direction, candleNum))
    {
        return false;
    }
  
    Print("فحص الشمعه[", candleNum, ":5] تحقق الشرط الثالث");
    Print("تحققت جميع الشروط للشمعة T1");
  
    return true;
}

//+------------------------------------------------------------------+
//| Check internal touch condition for H1                          |
//+------------------------------------------------------------------+
bool CheckInternalTouchH1(int t1Index, int direction, int candleNum)
{
    // Find B1 index
    int b1Index = FindH1CandleIndexByTime(B1Time);
    if(b1Index < 0)
    {
        Print("فحص الشمعه[", candleNum, ":5] خطأ: لم يتم العثور على B1");
        return false;
    }
  
    Print("فحص الشمعه[", candleNum, ":5] فحص اللمس الداخلي من B1 (Index: ", b1Index, ") إلى T1 (Index: ", t1Index, ")");
  
    // Check all candles between B1 and T1 (including T1)
    for(int i = b1Index - 1; i >= t1Index; i--)
    {
        double candleHigh = iHigh(TradingSymbol, PERIOD_H1, i);
        double candleLow = iLow(TradingSymbol, PERIOD_H1, i);
      
        if(direction == 1) // Buy - check if any low touched B1 low
        {
            if(candleLow <= B1Low)
            {
                Print("فحص الشمعه[", candleNum, ":5] لم يتحقق الشرط الثالث - تم لمس قاع B1 في الشمعة Index: ", i);
                Print("Candle Low: ", candleLow, " <= B1 Low: ", B1Low);
                ResetB1();
                return false;
            }
        }
        else // Sell - check if any high touched B1 high
        {
            if(candleHigh >= B1High)
            {
                Print("فحص الشمعه[", candleNum, ":5] لم يتحقق الشرط الثالث - تم لمس قمة B1 في الشمعة Index: ", i);
                Print("Candle High: ", candleHigh, " >= B1 High: ", B1High);
                ResetB1();
                return false;
            }
        }
    }
  
    Print("فحص الشمعه[", candleNum, ":5] لم يحدث لمس داخلي - الشرط الثالث محقق");
    return true;
}

//+------------------------------------------------------------------+
//| Setup Fibonacci levels for H1                                  |
//+------------------------------------------------------------------+
void SetupFibonacci()
{
    if(T1Direction == 1) // Buy
    {
        FibLevel1 = B1High;
        FibLevel0 = B1Low;
    }
    else // Sell
    {
        FibLevel1 = B1Low;
        FibLevel0 = B1High;
    }
  
    double range = MathAbs(FibLevel1 - FibLevel0);
    CancelLevelMinus01 = FibLevel0 - 0.1 * range;
    CancelLevel423 = FibLevel0 + 4.23 * range;
  
    Print("تم رسم فيبوناتشي H1:");
    Print("المستوى 0: ", FibLevel0);
    Print("المستوى 1: ", FibLevel1);
    Print("مستوى الالغاء -0.1: ", CancelLevelMinus01);
    Print("مستوى الالغاء 4.23: ", CancelLevel423);
}

//+------------------------------------------------------------------+
//| Check cancellation levels                                       |
//+------------------------------------------------------------------+
void CheckCancellationLevels()
{
    symbolInfo.RefreshRates();
    double currentPrice = (symbolInfo.Bid() + symbolInfo.Ask()) / 2;
  
    if(currentPrice <= CancelLevelMinus01 || currentPrice >= CancelLevel423)
    {
        Print("تم لمس مستوى الالغاء - البحث عن B1 جديدة");
        Print("السعر الحالي: ", currentPrice);
        ResetAll();
    }
}

//+------------------------------------------------------------------+
//| Check if price touched B1 range (only once)                    |
//+------------------------------------------------------------------+
bool CheckB1Touch()
{
    symbolInfo.RefreshRates();
    double currentBid = symbolInfo.Bid();
    double currentAsk = symbolInfo.Ask();
  
    if(T1Direction == 1) // Buy - wait for touch of B1 high
    {
        if(currentBid <= B1High && currentAsk >= B1High)
        {
            return true;
        }
    }
    else // Sell - wait for touch of B1 low
    {
        if(currentBid <= B1Low && currentAsk >= B1Low)
        {
            return true;
        }
    }
  
    return false;
}

//+------------------------------------------------------------------+
//| Analyze M5 timeframe                                           |
//+------------------------------------------------------------------+
void AnalyzeM5()
{
    if(!B11Found)
    {
        SearchForB11();
    }
    else
    {
        SearchForT11();
    }
}

//+------------------------------------------------------------------+
//| Search for B11 candle on M5                                    |
//+------------------------------------------------------------------+
void SearchForB11()
{
    // Get last closed candle on M5
    long lastVolume = iVolume(TradingSymbol, PERIOD_M5, 1);
    datetime lastTime = iTime(TradingSymbol, PERIOD_M5, 1);
  
    if(lastVolume <= 0) return; // No data available
  
    // Calculate average volume of last 20 candles on M5
    double avgVolume = CalculateAverageVolume(PERIOD_M5, 20);
  
    if(avgVolume <= 0) return; // Cannot calculate average
  
    // Check if last candle volume is >= average volume
    if((double)lastVolume >= avgVolume)
    {
        B11Found = true;
        B11High = iHigh(TradingSymbol, PERIOD_M5, 1);
        B11Low = iLow(TradingSymbol, PERIOD_M5, 1);
        B11Volume = lastVolume;
        B11Time = lastTime;
        candleCount = 0;
      
        Print("تم ايجاد شمعة B11 حجمها ", (double)B11Volume, " > متوسط الحجم ", avgVolume);
    }
}

//+------------------------------------------------------------------+
//| Find candle index by time on M5                                |
//+------------------------------------------------------------------+
int FindM5CandleIndexByTime(datetime targetTime)
{
    int maxBars = iBars(TradingSymbol, PERIOD_M5);
  
    for(int i = 0; i < maxBars; i++)
    {
        datetime candleTime = iTime(TradingSymbol, PERIOD_M5, i);
        if(candleTime == targetTime)
        {
            return i;
        }
        if(candleTime < targetTime) // We've gone too far back
        {
            break;
        }
    }
    return -1; // Not found
}

//+------------------------------------------------------------------+
//| Search for T11 candle on M5                                    |
//+------------------------------------------------------------------+
void SearchForT11()
{
    candleCount++;
  
    if(candleCount > 5)
    {
        // Reset B11 and search again
        Print("لم يحدث اختراق خلال 5 شموع M5 - إعادة تعيين B11");
        ResetB11();
        return;
    }
  
    double candleClose = iClose(TradingSymbol, PERIOD_M5, 1); // Last closed candle
    double candleHigh = iHigh(TradingSymbol, PERIOD_M5, 1);
    double candleLow = iLow(TradingSymbol, PERIOD_M5, 1);
    bool validBreakout = false;
  
    if(T1Direction == 1) // Buy signal from H1
    {
        if(candleClose > B11High)
        {
            Print("فحص الشمعة [", candleCount, ":5] اغلاق مع الاتجاه");
            validBreakout = true;
        }
        else if(candleClose < B11Low)
        {
            Print("فحص الشمعة [", candleCount, ":5] اغلاق عكس الاتجاه");
            ResetB11();
            return;
        }
    }
    else // Sell signal from H1
    {
        if(candleClose < B11Low)
        {
            Print("فحص الشمعة [", candleCount, ":5] اغلاق مع الاتجاه");
            validBreakout = true;
        }
        else if(candleClose > B11High)
        {
            Print("فحص الشمعة [", candleCount, ":5] اغلاق عكس الاتجاه");
            ResetB11();
            return;
        }
    }
  
    if(validBreakout)
    {
        if(CheckT11Conditions())
        {
            T11Found = true;
            T11High = candleHigh;
            T11Low = candleLow;
            T11Close = candleClose;
            T11Volume = iVolume(TradingSymbol, PERIOD_M5, 1);
            T11Time = iTime(TradingSymbol, PERIOD_M5, 1);
          
            ExecuteTrade();
        }
    }
}

//+------------------------------------------------------------------+
//| Check T11 conditions                                           |
//+------------------------------------------------------------------+
bool CheckT11Conditions()
{
    long candleVolume = iVolume(TradingSymbol, PERIOD_M5, 1);
  
    // Check volume condition
    if(candleVolume >= B11Volume)
    {
        Print("فحص الشمعة [", candleCount, ":5] لن يتحقق شرط الحجم حجم الشمعه ", (double)candleVolume, " > حجم شمعه B11 ", (double)B11Volume);
        ResetB11();
        return false;
    }
  
    Print("فحص الشمعة [", candleCount, ":5] تحقق شرط الحجم حجم الشمعه ", (double)candleVolume, " < حجم شمعه B11 ", (double)B11Volume);
  
    // Check internal touch condition for M5
    if(!CheckM5InternalTouch())
    {
        Print("فحص الشمعة [", candleCount, ":5] لم يتحقق الشرط الثالث");
        ResetB11();
        return false;
    }
  
    Print("فحص الشمعة [", candleCount, ":5] تحقق الشرط الثالث");
    Print("تحققت جميع الشروط");
  
    return true;
}

//+------------------------------------------------------------------+
//| Check M5 internal touch condition                              |
//+------------------------------------------------------------------+
bool CheckM5InternalTouch()
{
    // Find B11 index
    int b11Index = FindM5CandleIndexByTime(B11Time);
    if(b11Index < 0) return false;
  
    // Check candles between B11 and T11 (including T11)
    for(int i = b11Index - 1; i >= 1; i--)
    {
        double candleHigh = iHigh(TradingSymbol, PERIOD_M5, i);
        double candleLow = iLow(TradingSymbol, PERIOD_M5, i);
      
        if(T1Direction == 1) // Buy
        {
            if(candleLow <= B11Low)
            {
                return false;
            }
        }
        else // Sell
        {
            if(candleHigh >= B11High)
            {
                return false;
            }
        }
    }
  
    return true;
}

//+------------------------------------------------------------------+
//| Execute trade order                                             |
//+------------------------------------------------------------------+
void ExecuteTrade(){
    // Setup M5 Fibonacci
    if(T1Direction == 1) // Buy
    {
        M5FibLevel1 = T11High; // T11 high
        M5FibLevel0 = B11Low;  // B11 low
    }
    else // Sell
    {
        M5FibLevel1 = T11Low;  // T11 low
        M5FibLevel0 = B11High; // B11 high
    }
  
    double range = M5FibLevel1 - M5FibLevel0;
    EntryLevel = M5FibLevel0 + 1 * range;
    StopLoss = M5FibLevel0 - 0.1 * MathAbs(range);
    TakeProfit = M5FibLevel0 + 6.8 * MathAbs(range);
  
    // Adjust levels based on direction
    if(T1Direction == -1) // Sell
    {
        StopLoss = M5FibLevel0 + 0.1 * MathAbs(range);
        TakeProfit = M5FibLevel0 - 6.8 * MathAbs(range);
    }
  
    Print("تم رسم فيبوناتشي M5:");
    Print("المستوى 0: ", M5FibLevel0);
    Print("المستوى 1: ", M5FibLevel1);
    Print("مستوى الدخول (0.618): ", EntryLevel);
    Print("وقف الخسارة: ", StopLoss);
    Print("جني الأرباح: ", TakeProfit);
  
    // Determine order type
    ENUM_ORDER_TYPE orderType;
    string orderComment = "Volume Breakout Bot";
  
    symbolInfo.RefreshRates();
   
    if(T1Direction == 1) // Buy
    {
        double currentAsk = symbolInfo.Ask();
        if(EntryLevel > currentAsk)
        {
            orderType = ORDER_TYPE_BUY_STOP;
        }
        else
        {
            orderType = ORDER_TYPE_BUY_LIMIT;
        }
        orderComment += " - BUY";
    }
    else // Sell
    {
        double currentBid = symbolInfo.Bid();
        if(EntryLevel < currentBid)
        {
            orderType = ORDER_TYPE_SELL_STOP;
        }
        else
        {
            orderType = ORDER_TYPE_SELL_LIMIT;
        }
        orderComment += " - SELL";
    }
  
    // Place the order
    bool result = trade.OrderOpen(
        TradingSymbol,
        orderType,
        LotSize,
        0, // Limit price (will be set by entry level)
        NormalizeDouble(EntryLevel, _Digits),
        NormalizeDouble(StopLoss, _Digits),
        NormalizeDouble(TakeProfit, _Digits),
        ORDER_TIME_GTC,
        0, // Expiration
        orderComment
    );
  
    if(result)
    {
        Print("تم وضع الأمر بنجاح - التذكرة: ", trade.ResultOrder());
        Print("نوع الأمر: ", (T1Direction == 1) ? "شراء" : "بيع");
        Print("سعر الدخول: ", EntryLevel);
        Print("وقف الخسارة: ", StopLoss);
        Print("جني الأرباح: ", TakeProfit);
        
    }
    else
    {
        Print("فشل في وضع الأمر - الخطأ: ", GetLastError());
        Print("وصف الخطأ: ", trade.ResultRetcodeDescription());
        
    }
  
    // Reset B11 only to search for new B11 from the same T11
   }