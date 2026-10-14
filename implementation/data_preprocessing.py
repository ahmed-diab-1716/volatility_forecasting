import pandas as pd
import numpy as np


INTERVAL_MINUTES = 10
MARKET_OPEN_MINUTE = 9 * 60 + 30
MARKET_CLOSE_MINUTE = 16 * 60


def load_data(file_path):
    df = pd.read_csv(file_path)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    df["date"] = df["datetime"].dt.date
    df["minute_of_day"] = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute

    return df


def calculate_returns(df):
    df = df.copy()

    df["previous_datetime"] = df.groupby("date")["datetime"].shift(1)
    df["previous_close"] = df.groupby("date")["Close"].shift(1)

    df["gap_minutes"] = (df["datetime"] - df["previous_datetime"]).dt.total_seconds() / 60

    df["log_return"] = np.log(df["Close"] / df["previous_close"])

    # Only use genuine one-minute intraday returns.
    df["valid_return"] = df["gap_minutes"] == 1
    df.loc[~df["valid_return"], "log_return"] = np.nan

    df["squared_return"] = df["log_return"] ** 2

    return df


def calculate_realised_variance(df, interval_minutes):
    df = df.copy()

    # A return timestamped 09:31 corresponds to (09:30, 09:31].
    df["return_start_minute"] = df["minute_of_day"] - 1

    # Keep only returns belonging to the regular trading session.
    in_session = (
        (df["return_start_minute"] >= MARKET_OPEN_MINUTE)
        & (df["minute_of_day"] <= MARKET_CLOSE_MINUTE)
    )

    df = df.loc[in_session].copy()

    # Assign returns to a fixed clock-time grid.
    df["block"] = (df["return_start_minute"] - MARKET_OPEN_MINUTE) // interval_minutes

    rv = df.groupby(["date", "block"]).agg(
        realised_variance=("squared_return", "sum"),
        observed_returns=("squared_return", "count")
    ).reset_index()

    # Interval boundaries come from the fixed grid, not from available observations.
    rv["interval_start_minute"] = MARKET_OPEN_MINUTE + rv["block"] * interval_minutes
    rv["interval_end_minute"] = rv["interval_start_minute"] + interval_minutes

    rv["interval_minutes"] = interval_minutes
    rv["expected_returns"] = interval_minutes

    # Do not allow blocks extending beyond the regular close.
    rv["full_session_block"] = rv["interval_end_minute"] <= MARKET_CLOSE_MINUTE

    # A valid observation requires every one-minute return in the block.
    rv["valid"] = (
        (rv["observed_returns"] == rv["expected_returns"])
        & rv["full_session_block"]
    )

    rv.loc[~rv["valid"], "realised_variance"] = np.nan

    return rv


def add_timestamps(rv):
    rv = rv.copy()

    date = pd.to_datetime(rv["date"])

    rv["interval_start"] = date + pd.to_timedelta(rv["interval_start_minute"], unit="m")
    rv["interval_end"] = date + pd.to_timedelta(rv["interval_end_minute"], unit="m")

    return rv


def add_log_variance(rv):
    rv = rv.copy()

    # log(RV) is our observed log integrated-variance quantity.
    # Zero RV observations cannot be logged.
    rv["log_realised_variance"] = np.nan

    positive_rv = rv["realised_variance"] > 0
    rv.loc[positive_rv, "log_realised_variance"] = np.log(
        rv.loc[positive_rv, "realised_variance"]
    )

    return rv


def preprocess_data(input_path, output_path, interval_minutes):
    df = load_data(input_path)
    df = calculate_returns(df)

    rv = calculate_realised_variance(df, interval_minutes)
    rv = add_timestamps(rv)
    rv = add_log_variance(rv)

    columns = [
        "date",
        "interval_start",
        "interval_end",
        "interval_minutes",
        "expected_returns",
        "observed_returns",
        "realised_variance",
        "log_realised_variance",
        "valid"
    ]

    rv = rv[columns]
    rv.to_csv(output_path, index=False)

    return rv


if __name__ == "__main__":
    input_path = "data/SPY_clean_1min.csv"
    output_path = "data/SPY_10min_rv.csv"

    rv = preprocess_data(input_path, output_path, INTERVAL_MINUTES)

    print("Processed data saved to:", output_path)
    print()
    print("Interval:", INTERVAL_MINUTES, "minutes")
    print("Total blocks:", len(rv))
    print("Valid blocks:", rv["valid"].sum())
    print("Invalid blocks:", (~rv["valid"]).sum())
    print("Zero-variance valid blocks:", ((rv["realised_variance"] == 0) & rv["valid"]).sum())