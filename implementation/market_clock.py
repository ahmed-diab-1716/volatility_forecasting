import pandas as pd


class MarketClock:

    def __init__(self, c_night, market_open="09:30", market_close="16:00"):
        if c_night < 0:
            raise ValueError("c_night must be non-negative")

        self.c_night = c_night
        self.market_open = pd.Timedelta(market_open + ":00")
        self.market_close = pd.Timedelta(market_close + ":00")


    def effective_time(self, start, end):
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        if end < start:
            raise ValueError("end must be after start")

        if end == start:
            return 0.0

        open_hours = self.open_hours(start, end)
        total_hours = (end - start).total_seconds() / 3600
        closed_hours = total_hours - open_hours

        return open_hours + self.c_night * closed_hours


    def open_hours(self, start, end):
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        total_open_hours = 0.0
        current_date = start.normalize()

        while current_date <= end.normalize():
            if current_date.weekday() < 5:
                session_open = current_date + self.market_open
                session_close = current_date + self.market_close

                overlap_start = max(start, session_open)
                overlap_end = min(end, session_close)

                if overlap_end > overlap_start:
                    total_open_hours += (overlap_end - overlap_start).total_seconds() / 3600

            current_date += pd.Timedelta(days=1)

        return total_open_hours