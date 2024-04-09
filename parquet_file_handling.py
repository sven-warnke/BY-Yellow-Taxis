import dataclasses
import pathlib as pl
import logging
from typing import List, Tuple

import pandas as pd

DATA_FOLDER = pl.Path(__file__).parent / 'data'
DEFINING_TIME_COLUMN = 'tpep_pickup_datetime'


@dataclasses.dataclass
class MonthIdentifier:
    year: int
    month: int

    def __post_init__(self):
        if self.month < 1 or self.month > 12:
            raise ValueError(f'Month must be between 1 and 12, got {self.month}')

    def __lt__(self, other: 'MonthIdentifier') -> bool:
        return (self.year, self.month) < (other.year, other.month)

    def __le__(self, other: 'MonthIdentifier') -> bool:
        return (self.year, self.month) <= (other.year, other.month)

    def to_tuple(self) -> Tuple[int, int]:
        return self.year, self.month

    def next_month(self) -> 'MonthIdentifier':
        return MonthIdentifier(self.year + (self.month + 1) // 13, (self.month % 12) + 1)


def parquet_file_name(month_id: MonthIdentifier) -> str:
    return f'yellow_tripdata_{month_id.year}-{month_id.month:02}.parquet'


def load_parquet_file(month_id: MonthIdentifier) -> pd.DataFrame:
    parquet_file = DATA_FOLDER / parquet_file_name(month_id)

    if not parquet_file.exists():
        raise FileNotFoundError(f'File {parquet_file} does not exist. Did you download the data?')

    df = pd.read_parquet(parquet_file, columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_distance'])

    if df.isnull().any().any():
        # haven't found Nan values in the data yet, but I want to know if they appear
        raise ValueError('Found NaN values in the loaded DataFrame')

    df['trip_length_time'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']

    return df


def filter_df_for_correct_time(df: pd.DataFrame, month_id: MonthIdentifier) -> pd.DataFrame:
    time_period = pd.Period(year=month_id.year, month=month_id.month, freq='M')
    buffer_slight_overlap = pd.Timedelta(1,
                                         unit='hour')  # parquet files sometimes include a few entries from the next or previous month

    start_limit = time_period.start_time - buffer_slight_overlap
    end_limit = time_period.end_time + buffer_slight_overlap

    out_of_range_indices = (df[DEFINING_TIME_COLUMN] < start_limit) | (
            df[DEFINING_TIME_COLUMN] >= end_limit)
    if out_of_range_indices.any():
        logging.warning(
            f'Found {out_of_range_indices.sum()} entries with invalid time in {month_id.year}-{month_id.month:02}. They will be removed.')

    return df[~out_of_range_indices]


def load_filtered_parquet_file(month_id: MonthIdentifier) -> pd.DataFrame:
    df = load_parquet_file(month_id)
    return filter_df_for_correct_time(df, month_id)


def daily_means_from_df(df: pd.DataFrame) -> pd.DataFrame:
    df['date'] = df['tpep_pickup_datetime'].dt.date

    # also collect count to be able to combine dataframes later
    return df.groupby('date')[['trip_distance', 'trip_length_time']].agg(['mean', 'count'])


def get_daily_means_for_month(month_id: MonthIdentifier) -> pd.DataFrame:
    filtered_df = load_filtered_parquet_file(month_id)
    return daily_means_from_df(filtered_df)


def get_months_in_range_inclusive(start: MonthIdentifier, end: MonthIdentifier) -> List[MonthIdentifier]:
    if start > end:
        raise ValueError('Start month must be before end month')

    months = []
    current = start
    while current <= end:
        months.append(current)
        current = current.next_month()

    return months


def get_daily_means_in_range(start: MonthIdentifier, end: MonthIdentifier) -> pd.DataFrame:
    months = get_months_in_range_inclusive(start, end)

    daily_means = {
        month_id.to_tuple(): get_daily_means_for_month(month_id)
        for month_id in months
    }

    daily_means_df = pd.concat(daily_means)
    return daily_means_df
