import pathlib as pl
import warnings

import pandas as pd

DATA_FOLDER = pl.Path(__file__).parent / 'data'
DEFINING_TIME_COLUMN = 'tpep_pickup_datetime'


def parquet_file_name(year: int, month: int) -> str:
    if month < 1 or month > 12:
        raise ValueError(f'Month must be between 1 and 12, got {month}')

    return f'yellow_tripdata_{year}-{month:02}.parquet'


def load_parquet_file(year: int, month: int) -> pd.DataFrame:
    parquet_file = DATA_FOLDER / parquet_file_name(year, month)

    if not parquet_file.exists():
        raise FileNotFoundError(f'File {parquet_file} does not exist. Did you download the data?')

    df = pd.read_parquet(parquet_file, columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_distance'])

    if df.isnull().any().any():
        # haven't found Nan values in the data yet, but I want to know if they appear
        raise ValueError('Found NaN values in the loaded DataFrame')

    df['trip_length_time'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']

    return df


def filter_df_for_correct_time(df: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    time_period = pd.Period(year=year, month=month, freq='M')

    invalid_indices = (df[DEFINING_TIME_COLUMN] < time_period.start_time) | (
            df[DEFINING_TIME_COLUMN] >= time_period.end_time)
    if invalid_indices.any():
        warnings.warn(
            f'Found {invalid_indices.sum()} entries with invalid time in {year}-{month:02}. They will be removed.')

    return df[~invalid_indices]


def load_filtered_parquet_file(year: int, month: int) -> pd.DataFrame:
    return filter_df_for_correct_time(load_parquet_file(year, month), year, month)


def daily_means_from_df(df: pd.DataFrame) -> pd.DataFrame:
    df['date'] = df['tpep_pickup_datetime'].dt.date
    return df.groupby('date', as_index=False)[['trip_distance', 'trip_length_time']].mean()
