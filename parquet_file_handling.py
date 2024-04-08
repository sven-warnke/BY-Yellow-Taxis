import dataclasses

import pathlib as pl
import warnings
from typing import List

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

    invalid_indices = (df[DEFINING_TIME_COLUMN] < time_period.start_time) | (
            df[DEFINING_TIME_COLUMN] >= time_period.end_time)
    if invalid_indices.any():
        warnings.warn(
            f'Found {invalid_indices.sum()} entries with invalid time in {month_id.year}-{month_id.month:02}. They will be removed.')

    return df[~invalid_indices]


def load_filtered_parquet_file(month_id: MonthIdentifier) -> pd.DataFrame:
    df = load_parquet_file(month_id)
    return filter_df_for_correct_time(df, month_id)


def daily_means_from_df(df: pd.DataFrame) -> pd.DataFrame:
    df['date'] = df['tpep_pickup_datetime'].dt.date
    return df.groupby('date', as_index=False)[['trip_distance', 'trip_length_time']].mean()


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
        current = MonthIdentifier(current.year + (current.month + 1) // 13, (current.month % 12) + 1)

    return months



def get_daily_means_in_range(start: MonthIdentifier, end: MonthIdentifier) -> pd.DataFrame:
    months = get_months_in_range_inclusive(start, end)

    daily_means = [
        get_daily_means_for_month(month_id)
        for month_id in months
    ]

    daily_means_df = pd.concat(daily_means)
    return daily_means_df

