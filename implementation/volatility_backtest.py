import numpy as np
import pandas as pd

from implementation.market_clock import MarketClock
from implementation.multi_ou_filter import MultiOUFilter
from implementation.volatility_forecaster import VolatilityForecaster


def backtest_model(
    process,
    observation_variance,
    rv_data,
    daily_rv,
    seasonal_mean,
    c_night,
    start_date,
    end_date,
    horizons
):
    clock = MarketClock(c_night=c_night)

    filter_model = MultiOUFilter(
        model=process,
        observation_variance=observation_variance
    )

    forecaster = VolatilityForecaster(
        filter_model=filter_model,
        clock=clock,
        seasonal_mean=seasonal_mean
    )

    rv_data = rv_data.sort_values("interval_start").copy()

    times = rv_data["interval_start"].to_numpy()
    interval_ends = rv_data["interval_end"].to_numpy()
    observations = rv_data["adjusted_log_rv"].to_numpy()
    dates = rv_data["interval_start"].dt.normalize().to_numpy()

    trading_dates = daily_rv["date"].to_numpy()
    target_lookup = daily_rv.set_index("date")

    trading_date_positions = {
        date: i for i, date in enumerate(trading_dates)
    }

    last_observation = (
        rv_data.groupby(rv_data["interval_start"].dt.normalize())["interval_start"]
        .max()
        .to_dict()
    )

    results = []
    previous_time = None
    maximum_horizon = max(horizons)

    for current_time, interval_end, observation, current_date in zip(
        times,
        interval_ends,
        observations,
        dates
    ):
        current_time = pd.Timestamp(current_time)
        interval_end = pd.Timestamp(interval_end)
        current_date = pd.Timestamp(current_date)

        if previous_time is not None:
            if previous_time.date() == current_time.date():
                delta_t = (current_time - previous_time).total_seconds() / 3600
            else:
                delta_t = clock.effective_time(previous_time, current_time)

            filter_model.predict(delta_t)

        filter_model.update(observation)
        previous_time = current_time

        if current_time != last_observation[current_date]:
            continue

        if current_date < start_date or current_date >= end_date:
            continue

        if current_date not in trading_date_positions:
            continue

        position = trading_date_positions[current_date]

        future_dates = trading_dates[
            position + 1:position + 1 + maximum_horizon
        ]

        if len(future_dates) < maximum_horizon:
            continue

        future_dates = [
            pd.Timestamp(date)
            for date in future_dates
        ]

        forecasts = forecaster.forecast_horizons(
            forecast_origin=interval_end,
            future_dates=future_dates,
            horizons=horizons
        )

        result = {
            "date": current_date,
            "c_night": c_night
        }

        for horizon in horizons:
            result[f"forecast_{horizon}d"] = forecasts[horizon]
            result[f"actual_{horizon}d"] = target_lookup.loc[
                current_date,
                f"target_{horizon}d"
            ]

        results.append(result)

    return pd.DataFrame(results)