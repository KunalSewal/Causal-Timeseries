"""
Download Sample Stock Data

Downloads historical stock prices from Yahoo Finance for testing.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime, timedelta


def download_stock_data(
    tickers: list,
    start_date: str = None,
    end_date: str = None,
    output_path: str = 'data/raw/stock_prices.csv',
):
    """
    Download stock price data from Yahoo Finance.
    
    Args:
        tickers: List of stock tickers
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_path: Output file path
    """
    # Default dates: last 5 years
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    print(f"Downloading data for {tickers}...")
    print(f"Period: {start_date} to {end_date}")
    
    try:
        # Download data
        data = yf.download(tickers, start=start_date, end=end_date)['Close']
        
        # Handle single ticker case
        if len(tickers) == 1:
            data = pd.DataFrame(data, columns=tickers)
        
        # Drop rows with missing data
        data = data.dropna()
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path)
        
        print(f"\n✅ Downloaded {len(data)} days of data")
        print(f"📁 Saved to: {output_path}")
        print(f"\nPreview:")
        print(data.head())
        print(f"\nShape: {data.shape}")
        
        return data
    
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Download stock price data')
    parser.add_argument(
        '--tickers',
        nargs='+',
        default=['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
        help='Stock tickers to download'
    )
    parser.add_argument(
        '--start',
        type=str,
        default=None,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end',
        type=str,
        default=None,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/stock_prices.csv',
        help='Output file path'
    )
    
    args = parser.parse_args()
    
    download_stock_data(
        tickers=args.tickers,
        start_date=args.start,
        end_date=args.end,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
