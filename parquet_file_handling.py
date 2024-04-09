import dataclasses
import pathlib as pl
import logging
from typing import List, Tuple
import collections

import pandas as pd

DATA_FOLDER = pl.Path(__file__).parent / 'data'

TIME_COLUMNS = ['tpep_pickup_datetime', 'tpep_dropoff_datetime']
DEFINING_TIME_COLUMN = 'tpep_pickup_datetime'

ASSUMED_ORIGIN_TZ = 'UTC'
NEW_YORK_TZ = 'US/Eastern'


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

    def start_timestamp(self) -> pd.Timestamp:
        return pd.Period(year=self.year, month=self.month, freq='M').start_time.tz_localize(ASSUMED_ORIGIN_TZ)

    def end_timestamp(self) -> pd.Timestamp:
        return pd.Period(year=self.year, month=self.month, freq='M').end_time.tz_localize(ASSUMED_ORIGIN_TZ)


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

    for time_column in TIME_COLUMNS:
        if time_column not in df.columns:
            raise ValueError(f'Column {time_column} not found in the loaded DataFrame')
        df[time_column] = df[time_column].dt.tz_localize(ASSUMED_ORIGIN_TZ)

    df['trip_length_time'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']

    return df


def filter_df_for_correct_time(df: pd.DataFrame, month_id: MonthIdentifier) -> pd.DataFrame:
    buffer_slight_overlap = pd.Timedelta(1,
                                         unit='hour')  # parquet files sometimes include a few entries from the next or previous month

    start_limit = month_id.start_timestamp() - buffer_slight_overlap
    end_limit = month_id.end_timestamp() + buffer_slight_overlap

    out_of_range_indices = (df[DEFINING_TIME_COLUMN] < start_limit) | (
            df[DEFINING_TIME_COLUMN] >= end_limit)
    if out_of_range_indices.any():
        logging.warning(
            f'Found {out_of_range_indices.sum()} entries with invalid time in {month_id.year}-{month_id.month:02}. They will be removed.')
        for i, row in df[out_of_range_indices].iterrows():
            logging.info(f'Invalid time at index {i}: {row[DEFINING_TIME_COLUMN]}')

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


def repair_slightly_overlapping_dfs(df_before: pd.DataFrame, month_before: MonthIdentifier, df_after: pd.DataFrame,
                                    month_after: MonthIdentifier) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # slightly overlapping means that the last index of one df is the same as the first index of the next df
    if df_before.empty or df_after.empty:
        return df_before, df_after

    indices_wrong_first_df = df_before.index
    if df_before.index:
        return df_before, df_after

    return pd.concat(dfs)


def get_daily_means_in_range(start: MonthIdentifier, end: MonthIdentifier) -> pd.DataFrame:
    months = get_months_in_range_inclusive(start, end)

    daily_means = [
            (month_id.to_tuple(), get_daily_means_for_month(month_id))
            for month_id in months
    ]

    daily_means_df = pd.concat([x for _, x in daily_means])
    return daily_means_df
