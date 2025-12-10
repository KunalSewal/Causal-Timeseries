"""
Financial Data Downloaders

Multi-source data ingestion with caching, validation, and preprocessing.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FinancialDataDownloader:
    """
    Download financial data from multiple sources with caching and error handling.

    Supports:
    - Stock market data (Yahoo Finance)
    - Cryptocurrency data (CoinGecko API - free)
    - Economic indicators (FRED API - free)
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize downloader.

        Args:
            cache_dir: Directory for caching downloaded data
        """
        self.cache_dir = cache_dir or Path("data/raw")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Data cache directory: {self.cache_dir}")

    def download_stocks(
        self,
        tickers: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = "1d",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Download stock market data from Yahoo Finance.

        Args:
            tickers: List of stock tickers (e.g., ['AAPL', 'MSFT'])
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            interval: Data interval ('1d', '1h', etc.)
            use_cache: Use cached data if available

        Returns:
            DataFrame with adjusted close prices
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        cache_file = self.cache_dir / f"stocks_{'_'.join(tickers)}_{start_date}_{end_date}.csv"

        if use_cache and cache_file.exists():
            logger.info(f"Loading cached data from {cache_file}")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)

        logger.info(f"Downloading {len(tickers)} stocks from {start_date} to {end_date}")

        data_frames = []
        for ticker in tqdm(tickers, desc="Downloading stocks"):
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date, interval=interval)

                if hist.empty:
                    logger.warning(f"No data for {ticker}")
                    continue

                # Use adjusted close price
                df = hist[['Close']].copy()
                df.columns = [ticker]
                data_frames.append(df)

            except Exception as e:
                logger.error(f"Error downloading {ticker}: {e}")

        if not data_frames:
            raise ValueError("No data downloaded successfully")

        # Merge all tickers
        result = pd.concat(data_frames, axis=1)
        result = result.dropna()  # Remove days where any stock is missing

        # Save to cache
        result.to_csv(cache_file)
        logger.info(f"Downloaded {len(result)} rows, {len(result.columns)} columns")
        logger.info(f"Cached to {cache_file}")

        return result

    def download_crypto(
        self,
        symbols: List[str],
        days: int = 365,
        vs_currency: str = "usd",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Download cryptocurrency data from CoinGecko (free API).

        Args:
            symbols: List of crypto symbols (e.g., ['bitcoin', 'ethereum'])
            days: Number of days of historical data
            vs_currency: Currency to price against
            use_cache: Use cached data if available

        Returns:
            DataFrame with crypto prices
        """
        cache_file = self.cache_dir / f"crypto_{'_'.join(symbols)}_{days}d.csv"

        if use_cache and cache_file.exists():
            logger.info(f"Loading cached crypto data from {cache_file}")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)

        logger.info(f"Downloading {len(symbols)} cryptocurrencies for {days} days")

        # CoinGecko API mapping
        coingecko_ids = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'BNB': 'binancecoin',
            'SOL': 'solana',
            'ADA': 'cardano',
            'XRP': 'ripple',
            'DOT': 'polkadot',
            'DOGE': 'dogecoin',
        }

        data_frames = []
        for symbol in tqdm(symbols, desc="Downloading crypto"):
            try:
                # Convert symbol to CoinGecko ID
                coin_id = coingecko_ids.get(symbol.upper(), symbol.lower())

                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
                params = {
                    'vs_currency': vs_currency,
                    'days': days,
                    'interval': 'daily'
                }

                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                # Extract prices
                prices = data.get('prices', [])
                if not prices:
                    logger.warning(f"No data for {symbol}")
                    continue

                df = pd.DataFrame(prices, columns=['timestamp', symbol.upper()])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                data_frames.append(df)

            except Exception as e:
                logger.error(f"Error downloading {symbol}: {e}")

        if not data_frames:
            raise ValueError("No crypto data downloaded successfully")

        result = pd.concat(data_frames, axis=1)
        result = result.dropna()

        result.to_csv(cache_file)
        logger.info(f"Downloaded {len(result)} rows, {len(result.columns)} columns")

        return result

    def download_economic_indicators(
        self,
        indicators: Dict[str, str],
        start_date: str,
        end_date: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Download economic indicators from FRED (Federal Reserve Economic Data).

        Note: Requires FRED API key (free). Set FRED_API_KEY environment variable.

        Args:
            indicators: Dict mapping names to FRED series IDs
                       e.g., {'GDP': 'GDP', 'Inflation': 'CPIAUCSL'}
            start_date: Start date
            end_date: End date
            use_cache: Use cached data

        Returns:
            DataFrame with economic indicators
        """
        try:
            from fredapi import Fred
            import os

            api_key = os.getenv('FRED_API_KEY')
            if not api_key:
                logger.warning("FRED_API_KEY not set. Skipping economic indicators.")
                logger.info("Get free API key at: https://fred.stlouisfed.org/docs/api/api_key.html")
                return pd.DataFrame()

            cache_file = self.cache_dir / f"economic_{start_date}_{end_date}.csv"

            if use_cache and cache_file.exists():
                logger.info(f"Loading cached economic data from {cache_file}")
                return pd.read_csv(cache_file, index_col=0, parse_dates=True)

            fred = Fred(api_key=api_key)
            data_frames = []

            for name, series_id in tqdm(indicators.items(), desc="Downloading economic data"):
                try:
                    series = fred.get_series(series_id, start_date, end_date)
                    df = series.to_frame(name)
                    data_frames.append(df)
                except Exception as e:
                    logger.error(f"Error downloading {name}: {e}")

            if not data_frames:
                return pd.DataFrame()

            result = pd.concat(data_frames, axis=1)
            result = result.dropna()

            result.to_csv(cache_file)
            logger.info(f"Downloaded {len(result)} rows of economic data")

            return result

        except ImportError:
            logger.warning("fredapi not installed. Install with: pip install fredapi")
            return pd.DataFrame()

    def download_faang_tech_bundle(
        self,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        include_indices: bool = True,
    ) -> pd.DataFrame:
        """
        Download comprehensive tech stock bundle for analysis.

        Includes:
        - FAANG stocks
        - Major tech leaders
        - Market indices (if include_indices=True)

        Args:
            start_date: Start date
            end_date: End date
            include_indices: Include SPY, QQQ, VIX

        Returns:
            DataFrame with all stock data
        """
        tickers = [
            # FAANG
            'AAPL', 'AMZN', 'META', 'NFLX', 'GOOGL',
            # Tech Leaders
            'MSFT', 'NVDA', 'TSLA', 'AMD', 'INTC',
        ]

        if include_indices:
            tickers.extend(['SPY', 'QQQ', '^VIX'])

        return self.download_stocks(tickers, start_date, end_date)


def get_sample_datasets() -> Dict[str, pd.DataFrame]:
    """
    Download all sample datasets for the project.

    Returns:
        Dictionary with dataset names and DataFrames
    """
    downloader = FinancialDataDownloader()

    datasets = {}

    # 1. Tech stocks (5 years)
    logger.info("Downloading tech stocks dataset...")
    datasets['tech_stocks'] = downloader.download_faang_tech_bundle(
        start_date=(datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
    )

    # 2. Cryptocurrency (3 years)
    logger.info("Downloading cryptocurrency dataset...")
    try:
        datasets['crypto'] = downloader.download_crypto(
            symbols=['BTC', 'ETH', 'BNB', 'SOL', 'ADA'],
            days=3*365
        )
    except Exception as e:
        logger.error(f"Failed to download crypto: {e}")

    return datasets


if __name__ == "__main__":
    # Test data download
    logging.basicConfig(level=logging.INFO)

    downloader = FinancialDataDownloader()

    # Download tech stocks
    print("\n=== Downloading Tech Stocks ===")
    stocks = downloader.download_faang_tech_bundle()
    print(f"Shape: {stocks.shape}")
    print(stocks.head())
    print(f"\nDate range: {stocks.index.min()} to {stocks.index.max()}")

    # Download crypto
    print("\n=== Downloading Cryptocurrency ===")
    crypto = downloader.download_crypto(['BTC', 'ETH', 'SOL'], days=365)
    print(f"Shape: {crypto.shape}")
    print(crypto.head())
