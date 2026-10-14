import numpy as np
import pandas as pd


def covariance_at_exact_lag(data, lag, same_day=False, same_clock_time=False):
    current = data[["interval_start", "adjusted_log_rv"]].copy()

    if same_clock_time:
        current["clock_time"] = current["interval_start"].dt.strftime("%H:%M")

    future = current.copy()
    future["interval_start"] = future["interval_start"] - lag
    future = future.rename(columns={"adjusted_log_rv": "future_log_rv"})

    merge_columns = ["interval_start"]

    if same_clock_time:
        merge_columns.append("clock_time")

    pairs = current.merge(future, on=merge_columns)

    if same_day:
        future_time = pairs["interval_start"] + lag
        pairs = pairs[pairs["interval_start"].dt.date == future_time.dt.date]

    if len(pairs) == 0:
        return np.nan, 0

    covariance = np.mean(pairs["adjusted_log_rv"] * pairs["future_log_rv"])

    return covariance, len(pairs)


def estimate_covariance_data(data, max_intraday_hours=6, max_interday_days=30):
    results = []

    for minutes in range(10, int(max_intraday_hours * 60) + 1, 10):
        covariance, pairs = covariance_at_exact_lag(
            data,
            pd.Timedelta(minutes=minutes),
            same_day=True
        )

        results.append({
            "effective_hours": minutes / 60,
            "covariance": covariance,
            "pairs": pairs,
            "source": "Intraday"
        })

    for days in range(1, max_interday_days + 1):
        covariance, pairs = covariance_at_exact_lag(
            data,
            pd.Timedelta(days=days),
            same_clock_time=True
        )

        results.append({
            "effective_hours": days * 6.5,
            "covariance": covariance,
            "pairs": pairs,
            "source": "Interday"
        })

    covariance_data = pd.DataFrame(results)
    covariance_data = covariance_data.dropna(subset=["covariance"])
    covariance_data = covariance_data.sort_values("effective_hours").reset_index(drop=True)

    return covariance_data