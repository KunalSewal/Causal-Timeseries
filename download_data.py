"""
Quick Data Download Script - Downloads real financial data
"""

import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import yfinance as yf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_tech_stocks():
    """Download FAANG + Tech stocks (5 years)."""
    tickers = [
        'AAPL', 'AMZN', 'META', 'NFLX', 'GOOGL',  # FAANG
        'MSFT', 'NVDA', 'TSLA', 'AMD', 'INTC',     # Tech Leaders
        'SPY', 'QQQ', '^VIX'                        # Indices
    ]
    
    start_date = "2020-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"Downloading {len(tickers)} stocks from {start_date} to {end_date}")
    
    data_frames = []
    for ticker in tqdm(tickers, desc="Downloading"):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date, interval='1d')
            
            if hist.empty:
                logger.warning(f"No data for {ticker}")
                continue
            
            df = hist[['Close']].copy()
            df.columns = [ticker]
            data_frames.append(df)
            
        except Exception as e:
            logger.error(f"Error with {ticker}: {e}")
    
    if not data_frames:
        raise ValueError("No data downloaded!")
    
    result = pd.concat(data_frames, axis=1)
    result = result.dropna(how='all')
    result = result.fillna(method='ffill').fillna(method='bfill')
    
    # Save
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "tech_stocks_2020_2025.csv")
    
    logger.info(f"✓ Downloaded {len(result)} days × {len(result.columns)} stocks")
    logger.info(f"✓ Date range: {result.index.min().date()} to {result.index.max().date()}")
    logger.info(f"✓ Missing values: {result.isna().sum().sum()}")
    
    return result


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DOWNLOADING REAL FINANCIAL DATA")
    print("="*70 + "\n")
    
    stocks = download_tech_stocks()
    
    print(f"\nPreview:\n{stocks.head()}\n")
    print(f"\nSummary Statistics:\n{stocks.describe()}\n")
    
    print("="*70)
    print("DOWNLOAD COMPLETE!")
    print("="*70)
    print(f"\nFile: data/processed/tech_stocks_2020_2025.csv")
    print(f"Shape: {stocks.shape[0]} rows × {stocks.shape[1]} columns")
