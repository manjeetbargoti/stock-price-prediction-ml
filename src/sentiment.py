"""
src/sentiment.py
-----------------
SentimentAnalyzer — generates daily sentiment scores from financial
news headlines using VADER (NLTK).  Falls back to zero scores when
no external news API is configured, so the rest of the pipeline
continues to work out of the box.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import json, re

import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.logger import get_logger

log = get_logger(__name__)

# Download VADER lexicon once
try:
    import nltk
    nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _VADER_AVAILABLE = True
except Exception:
    _VADER_AVAILABLE = False
    log.warning("VADER not available. Sentiment will default to zero.")


class SentimentAnalyzer:
    """
    Compute daily sentiment scores from news headlines.

    Usage
    -----
    analyzer = SentimentAnalyzer()

    # Option A: provide a DataFrame with 'date' and 'headline' columns
    daily_df = analyzer.from_dataframe(news_df)

    # Option B: provide a list of (date_str, headline) tuples
    daily_df = analyzer.from_tuples(headline_tuples)

    # Option C: synthetic zeros (useful when no news data is available)
    daily_df = analyzer.make_zero_scores(date_index)
    """

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer() if _VADER_AVAILABLE else None

    # ── Public API ────────────────────────────────────────────────────────────

    def from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : DataFrame with at minimum columns ['date', 'headline']
             'date' can be any parseable date string or datetime.

        Returns
        -------
        DataFrame indexed by date with columns:
            sentiment_mean, sentiment_std, news_count,
            positive_ratio, negative_ratio
        """
        if not _VADER_AVAILABLE:
            log.warning("VADER unavailable — returning zero scores.")
            return self.make_zero_scores(pd.to_datetime(df["date"].unique()))

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["headline"])
        df["headline"] = df["headline"].astype(str).apply(self._clean_text)

        scores = df["headline"].apply(self._score)
        df["compound"] = scores.apply(lambda s: s["compound"])
        df["positive"] = scores.apply(lambda s: s["pos"])
        df["negative"] = scores.apply(lambda s: s["neg"])

        daily = (
            df.groupby("date")
            .agg(
                sentiment_mean = ("compound", "mean"),
                sentiment_std  = ("compound", "std"),
                news_count     = ("compound", "count"),
                positive_ratio = ("positive", "mean"),
                negative_ratio = ("negative", "mean"),
            )
            .fillna(0)
            .round(4)
        )
        log.info(
            "Sentiment computed for %d trading days (mean=%.3f).",
            len(daily), daily["sentiment_mean"].mean(),
        )
        return daily

    def from_tuples(self, tuples: list[tuple]) -> pd.DataFrame:
        """Convenience wrapper: list of (date_str, headline) tuples."""
        df = pd.DataFrame(tuples, columns=["date", "headline"])
        return self.from_dataframe(df)

    @staticmethod
    def make_zero_scores(date_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Return a DataFrame of zeros for every date in the index.
        Used as a neutral placeholder when no news data is available.
        """
        return pd.DataFrame(
            {
                "sentiment_mean": 0.0,
                "sentiment_std":  0.0,
                "news_count":     0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
            },
            index=pd.to_datetime(date_index),
        )

    def score_text(self, text: str) -> dict:
        """Score a single piece of text."""
        if not _VADER_AVAILABLE:
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
        return self.sia.polarity_scores(self._clean_text(text))

    # ── Private ───────────────────────────────────────────────────────────────

    def _score(self, text: str) -> dict:
        try:
            return self.sia.polarity_scores(text)
        except Exception:
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}

    @staticmethod
    def _clean_text(text: str) -> str:
        """Minimal cleaning: strip HTML tags, collapse whitespace."""
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


# ── Stand-alone demo ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_headlines = [
        ("2023-01-03", "Reliance Q3 profit surges 20%, beats analyst expectations"),
        ("2023-01-03", "Broader markets rally as FII buying continues"),
        ("2023-01-04", "Rupee falls to all-time low amid global dollar strength"),
        ("2023-01-04", "RBI holds rates; governor signals cautious outlook"),
        ("2023-01-05", "IT sector faces headwinds as US recession fears mount"),
    ]
    analyzer = SentimentAnalyzer()
    result   = analyzer.from_tuples(sample_headlines)
    print(result)
