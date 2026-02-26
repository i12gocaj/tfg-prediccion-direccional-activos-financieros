import os
import time
from datetime import datetime

import pandas as pd
import requests
import yfinance as yf

class DataDownloader:
    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def download_financial_data(self, tickers, start_date, end_date):
        # Baja datos de Yahoo Finance
        print(f"Downloading data for {tickers} from {start_date} to {end_date}...")
        success = True
        for ticker in tickers:
            try:
                # Fijar parámetros explícitos para reproducibilidad (y evitar cambios por defecto).
                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    progress=False,
                )
                if not df.empty:
                    # Guardar
                    filename = f"{ticker.replace('-', '_')}_history.csv"
                    filepath = os.path.join(self.data_dir, filename)
                    df.to_csv(filepath)
                    print(f"Saved {ticker} data to {filepath}")
                else:
                    print(f"No data found for {ticker}")
                    success = False
            except Exception as e:
                print(f"Error downloading {ticker}: {e}")
                success = False
        return success

    def _fetch_gdelt_timeline(self, query, mode, start_dt, end_dt, max_retries=3):
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            "query": query,
            "mode": mode,
            "format": "json",
            "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
            "enddatetime": end_dt.strftime("%Y%m%d%H%M%S")
        }
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(url, params=params, timeout=60)
                response.raise_for_status()
                try:
                    data = response.json()
                except ValueError:
                    preview = response.text[:200].strip().replace("\n", " ")
                    raise ValueError(f"GDELT non-JSON response: {preview}")
                timeline = data.get("timeline", [])
                if not timeline:
                    return pd.DataFrame()
                rows = timeline[0].get("data", [])
                if not rows:
                    return pd.DataFrame()
                df = pd.DataFrame(rows)
                df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%SZ", errors="coerce")
                df = df.dropna(subset=["date"]).set_index("date").sort_index()
                return df
            except Exception as e:
                if attempt >= max_retries:
                    raise e
                time.sleep(5 * attempt)
        return pd.DataFrame()

    def download_gdelt_sentiment(self, query="bitcoin", start_date=None, end_date=None,
                                 output_filename="sentiment_BTC_gdelt.csv"):
        # Descarga sentimiento de noticias (GDELT - Average Tone)
        if not start_date or not end_date:
            print("start_date/end_date required for GDELT sentiment.")
            return False

        print(f"Downloading GDELT tone for '{query}'...")
        tone_frames = []
        volume_frames = []

        try:
            for chunk_start, chunk_end in self._iter_year_ranges(start_date, end_date):
                tone_df = self._fetch_gdelt_timeline(query, "timelinetone", chunk_start, chunk_end)
                if not tone_df.empty:
                    tone_df = tone_df.rename(columns={"value": "Sentiment_Index"})
                    tone_frames.append(tone_df)

                volume_df = self._fetch_gdelt_timeline(query, "timelinevol", chunk_start, chunk_end)
                if not volume_df.empty:
                    volume_df = volume_df.rename(columns={"value": "Sentiment_Volume"})
                    volume_frames.append(volume_df)

                time.sleep(2)

            if not tone_frames:
                print(f"No GDELT tone data for '{query}'.")
                return False

            tone = pd.concat(tone_frames)
            tone = tone[~tone.index.duplicated(keep="last")].sort_index()

            if volume_frames:
                volume = pd.concat(volume_frames)
                volume = volume[~volume.index.duplicated(keep="last")].sort_index()
                tone = tone.join(volume, how="left")

            filepath = os.path.join(self.data_dir, output_filename)
            tone.to_csv(filepath)
            print(f"Saved GDELT sentiment to {filepath}")
            return True
        except Exception as e:
            print(f"Error downloading GDELT sentiment: {e}")
            return False

    def _iter_year_ranges(self, start_date, end_date):
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        year = start_dt.year
        while year <= end_dt.year:
            chunk_start = datetime(year, 1, 1)
            chunk_end = datetime(year, 12, 31)
            if year == start_dt.year:
                chunk_start = start_dt
            if year == end_dt.year:
                chunk_end = end_dt
            yield chunk_start, chunk_end
            year += 1

    def download_all(self, tickers=None, start_date="2020-11-01", end_date="2025-11-30"):
        # Descarga todo
        if tickers is None:
            tickers = ["BTC-USD", "TSLA", "NVDA"]
        success = self.download_financial_data(tickers, start_date, end_date)
        # Sentimiento especifico por activo (GDELT)
        success &= self.download_gdelt_sentiment(
            query="(bitcoin OR BTC)",
            start_date=start_date,
            end_date=end_date,
            output_filename="sentiment_BTC_gdelt.csv"
        )
        success &= self.download_gdelt_sentiment(
            query="(Tesla OR TSLA)",
            start_date=start_date,
            end_date=end_date,
            output_filename="sentiment_TSLA_gdelt.csv"
        )
        success &= self.download_gdelt_sentiment(
            query="(Nvidia OR NVDA)",
            start_date=start_date,
            end_date=end_date,
            output_filename="sentiment_NVDA_gdelt.csv"
        )
        return success

if __name__ == "__main__":
    downloader = DataDownloader()
    ok = downloader.download_all()
    if not ok:
        raise SystemExit(1)
