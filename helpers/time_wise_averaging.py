import abc

import pandas as pd
from helpers import constants as c


class TimeWiseAverager(abc.ABC):
    """
    As per the definition in the coding challenge, there are already 2 ways mentioned of calculating the mean of the
    trip distance and trip length in minutes over different time windows. This class is an abstract class that defines
    the interface for these sort of calculations in general.
    """

    @abc.abstractmethod
    def calculate_mean(self, daily_means_df: pd.DataFrame) -> pd.DataFrame:
        pass
    
    @abc.abstractmethod
    def name(self):
        """
        Used to describe method in the report 
        """
        pass

    @staticmethod
    def _prepare_df_for_grouping_operations(
        daily_means_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if daily_means_df.index.name != c.DATE_COLUMN:
            raise ValueError("Index must be date")
        daily_means_df = daily_means_df.sort_index().reset_index()
        daily_means_df[c.DATE_COLUMN] = pd.to_datetime(daily_means_df[c.DATE_COLUMN])
        daily_means_df[c.LENGTH_IN_MINS_COLUMN] = (
            daily_means_df[c.TIME_LENGTH_COLUMN].dt.total_seconds() / 60
        )
        return daily_means_df


class RollingMean(TimeWiseAverager):
    def __init__(self, window: str = "45D"):
        self.window = window

    def calculate_mean(self, daily_means_df: pd.DataFrame) -> pd.DataFrame:
        daily_means_df = TimeWiseAverager._prepare_df_for_grouping_operations(
            daily_means_df
        )
        return daily_means_df.rolling(self.window, on=c.DATE_COLUMN)[
            [c.DISTANCE_COLUMN, c.LENGTH_IN_MINS_COLUMN]
        ].mean()
    
    def name(self):
        return f"Rolling mean {self.window}"


class MonthlyMean(TimeWiseAverager):
    def calculate_mean(self, daily_means_df: pd.DataFrame) -> pd.DataFrame:
        daily_means_df = TimeWiseAverager._prepare_df_for_grouping_operations(
            daily_means_df
        )
        monthly_means = (
            daily_means_df.groupby(pd.Grouper(key=c.DATE_COLUMN, freq="ME"))[
                [c.DISTANCE_COLUMN, c.LENGTH_IN_MINS_COLUMN]
            ]
            .mean()
            .reset_index()
        )
        return monthly_means
    
    def name(self):
        return "Monthly mean"
