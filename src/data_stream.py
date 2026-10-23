import datetime
import itertools
import random
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

__all__ = [
    "Transaction",
    "TransactionStream",
    "stream_transactions",
    "sample_transactions",
]


@dataclass(frozen=True)
class Transaction:
    """A single financial transaction record.

    Attributes
    ----------
    transaction_id : int
        Unique identifier for the transaction.
    user_id : int
        Identifier of the user who performed the transaction.
    amount : float
        Transaction amount in US dollars.
    timestamp : datetime.datetime
        Time at which the transaction occurred.
    merchant : str
        Name of the merchant.
    category : str
        High‑level category of the purchase (e.g., “groceries”, “travel”).
    is_fraud : bool
        Ground‑truth label indicating whether the transaction is fraudulent.
    """

    transaction_id: int
    user_id: int
    amount: float
    timestamp: datetime.datetime
    merchant: str
    category: str
    is_fraud: bool

    def to_dict(self) -> Dict[str, Union[int, float, str, bool, datetime.datetime]]:
        """Return a shallow ``dict`` representation suitable for pandas."""
        return asdict(self)

    def to_series(self) -> pd.Series:
        """Return a ``pandas.Series`` representation."""
        return pd.Series(self.to_dict())


class TransactionStream(Iterable[Dict[str, Union[int, float, str, bool, datetime.datetime]]]):
    """Infinite generator that yields realistic synthetic transaction records.

    The stream is deterministic when a ``seed`` is supplied. It simulates a
    continuous flow of transactions, incrementing timestamps by a configurable
    ``time_step_seconds`` value. Fraudulent transactions are injected according
    to ``fraud_rate``.

    Parameters
    ----------
    num_users : int, default 1000
        Number of distinct users that can appear in the stream.
    start_time : datetime.datetime, optional
        Timestamp of the first transaction. If ``None`` the current UTC time is used.
    time_step_seconds : float, default 1.0
        Average number of seconds between consecutive transactions. The actual
        interval is drawn from an exponential distribution to mimic a Poisson
        process.
    fraud_rate : float, default 0.01
        Proportion of transactions that are fraudulent (between 0 and 1).
    seed : int, optional
        Seed for the internal random number generator to ensure reproducibility.
    merchants : list[str], optional
        Custom list of merchant names. If ``None`` a default list is used.
    categories : list[str], optional
        Custom list of transaction categories. If ``None`` a default list is used.
    """

    _DEFAULT_MERCHANTS = [
        "Amazon",
        "Walmart",
        "Target",
        "Starbucks",
        "Apple Store",
        "Google Play",
        "Netflix",
        "Uber",
        "Airbnb",
        "Best Buy",
    ]

    _DEFAULT_CATEGORIES = [
        "electronics",
        "groceries",
        "entertainment",
        "transportation",
        "travel",
        "food & beverage",
        "services",
        "health",
        "education",
        "misc",
    ]

    def __init__(
        self,
        num_users: int = 1000,
        start_time: Optional[datetime.datetime] = None,
        time_step_seconds: float = 1.0,
        fraud_rate: float = 0.01,
        seed: Optional[int] = None,
        merchants: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
    ) -> None:
        if num_users <= 0:
            raise ValueError("num_users must be a positive integer")
        if not (0.0 <= fraud_rate <= 1.0):
            raise ValueError("fraud_rate must be between 0 and 1")
        if time_step_seconds <= 0:
            raise ValueError("time_step_seconds must be positive")

        self.num_users = num_users
        self.current_time = start_time or datetime.datetime.utcnow()
        self.time_step_seconds = time_step_seconds
        self.fraud_rate = fraud_rate
        self.merchants = merchants or self._DEFAULT_MERCHANTS
        self.categories = categories or self._DEFAULT_CATEGORIES

        # Use numpy's Generator for reproducibility and vectorised draws
        self.rng = np.random.default_rng(seed)

        # Internal counter for transaction IDs
        self._tx_counter = itertools.count(1)

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, Union[int, float, str, bool, datetime.datetime]]:
        """Generate the next transaction record.

        Returns
        -------
        dict
            Mapping of column names to values.
        """
        # Increment timestamp using an exponential inter‑arrival time
        delta_seconds = self.rng.exponential(self.time_step_seconds)
        self.current_time += datetime.timedelta(seconds=float(delta_seconds))

        transaction_id = next(self._tx_counter)
        user_id = self.rng.integers(1, self.num_users + 1)

        # Transaction amount follows a log‑normal distribution (typical for spend data)
        amount = float(
            max(
                0.01,
                self.rng.lognormal(mean=3.0, sigma=1.0),
            )
        )  # enforce a minimum of $0.01

        merchant = self.rng.choice(self.merchants)
        category = self.rng.choice(self.categories)

        is_fraud = self.rng.random() < self.fraud_rate

        tx = Transaction(
            transaction_id=transaction_id,
            user_id=user_id,
            amount=amount,
            timestamp=self.current_time,
            merchant=merchant,
            category=category,
            is_fraud=is_fraud,
        )
        return tx.to_dict()


def stream_transactions(
    *,
    num_users: int = 1000,
    start_time: Optional[datetime.datetime] = None,
    time_step_seconds: float = 1.0,
    fraud_rate: float = 0.01,
    seed: Optional[int] = None,
    merchants: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
) -> Iterable[Dict[str, Union[int, float, str, bool, datetime.datetime]]]:
    """Convenient factory that returns an iterator over synthetic transactions.

    This function mirrors the arguments of :class:`TransactionStream` and is
    provided for a functional programming style.

    Yields
    ------
    dict
        Transaction records as plain dictionaries.
    """
    return iter(
        TransactionStream(
            num_users=num_users,
            start_time=start_time,
            time_step_seconds=time_step_seconds,
            fraud_rate=fraud_rate,
            seed=seed,
            merchants=merchants,
            categories=categories,
        )
    )


def sample_transactions(
    n: int,
    *,
    num_users: int = 1000,
    start_time: Optional[datetime.datetime] = None,
    time_step_seconds: float = 1.0,
    fraud_rate: float = 0.01,
    seed: Optional[int] = None,
    merchants: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Generate a finite ``pandas.DataFrame`` containing ``n`` synthetic transactions.

    Parameters
    ----------
    n : int
        Number of rows to generate.
    **kwargs
        Forwarded to :func:`stream_transactions`.

    Returns
    -------
    pandas.DataFrame
        DataFrame with ``n`` rows and the columns defined by :class:`Transaction`.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")

    stream = stream_transactions(
        num_users=num_users,
        start_time=start_time,
        time_step_seconds=time_step_seconds,
        fraud_rate=fraud_rate,
        seed=seed,
        merchants=merchants,
        categories=categories,
    )
    rows = [next(stream) for _ in range(n)]
    return pd.DataFrame(rows)