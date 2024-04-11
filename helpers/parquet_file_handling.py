import dataclasses
import datetime
import logging
import pathlib as pl
import urllib.request
import urllib.error
from typing import List, Tuple, Dict

import pandas as pd
import pyarrow.parquet

DATA_FOLDER = pl.Path(__file__).parent.parent / "data"

TIME_COLUMNS = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
DEFINING_TIME_COLUMN = "tpep_pickup_datetime"

ASSUMED_ORIGIN_TZ = "UTC"
NEW_YORK_TZ = "US/Eastern"

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclasses.dataclass
class MonthIdentifier:
    """
    Parquet files are named after the month and year they contain data for. This class is used to identify a month and
    its corresponding parquet file. It also contains utility functions for handling monthly parquet files.
    """

    year: int
    month: int

    def __post_init__(self):
        if self.month < 1 or self.month > 12:
            raise ValueError(f"Month must be between 1 and 12, got {self.month}")

    def __lt__(self, other: "MonthIdentifier") -> bool:
        return (self.year, self.month) < (other.year, other.month)

    def __le__(self, other: "MonthIdentifier") -> bool:
        return (self.year, self.month) <= (other.year, other.month)

    def next_month(self) -> "MonthIdentifier":
        return MonthIdentifier(
            self.year + (self.month + 1) // 13, (self.month % 12) + 1
        )

    def start_timestamp(self) -> pd.Timestamp:
        return pd.Period(
            year=self.year, month=self.month, freq="M"
        ).start_time.tz_localize(ASSUMED_ORIGIN_TZ)

    def end_timestamp(self) -> pd.Timestamp:
        return pd.Period(
            year=self.year, month=self.month, freq="M"
        ).end_time.tz_localize(ASSUMED_ORIGIN_TZ)

    def first_day_of_month(self) -> datetime.date:
        return datetime.date(self.year, self.month, 1)

    def last_day_of_month(self) -> datetime.date:
        next_month = self.next_month()
        return datetime.date(next_month.year, next_month.month, 1) - datetime.timedelta(
            days=1
        )


def parquet_file_name(month_id: MonthIdentifier) -> str:
    return f"yellow_tripdata_{month_id.year}-{month_id.month:02}.parquet"


def is_url_valid(url: str) -> bool:
    try:
        if urllib.request.urlopen(url).code == 200:
            return True
        else:
            return False
    except urllib.error.HTTPError:
        return False


def expected_parquet_file_url(month_id: MonthIdentifier) -> str:
    return (
        f"https://d37ci6vzurychx.cloudfront.net/trip-data/{parquet_file_name(month_id)}"
    )


def acquire_parquet_file(month_id: MonthIdentifier) -> pl.Path:
    url = expected_parquet_file_url(month_id)
    target = DATA_FOLDER / parquet_file_name(month_id)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        LOGGER.info(f"File {target} already exists, skipping download")
        return target

    LOGGER.info(f"Downloading {url} to {target}")
    urllib.request.urlretrieve(url, filename=target)
    if not target.exists():
        raise ValueError(f"Failed to download {url} to {target}")
    return target


@dataclasses.dataclass
class ColumnMapping:
    """
    Class to map column names from different schemas to a common schema. The class is used to load parquet files with different schemas.
    This is necessary because the column names in the parquet files can vary between months.
    """

    pickup_time: str
    dropoff_time: str
    distance: str

    def __post_init__(self):
        if self.pickup_time == self.dropoff_time:
            raise ValueError("Pickup and dropoff time must be different")
        if self.pickup_time == self.distance:
            raise ValueError("Pickup time and distance must be different")
        if self.dropoff_time == self.distance:
            raise ValueError("Dropoff time and distance must be different")

    def mapping_dict(self) -> Dict[str, str]:
        return {
            self.pickup_time: "tpep_pickup_datetime",
            self.dropoff_time: "tpep_dropoff_datetime",
            self.distance: "trip_distance",
        }

    def matches_columns(self, columns: List[str]) -> bool:
        return (
            self.pickup_time in columns
            and self.dropoff_time in columns
            and self.distance in columns
        )

    def column_names(self) -> List[str]:
        return [self.pickup_time, self.dropoff_time, self.distance]


def get_column_mapping(parquet_file: pl.Path) -> ColumnMapping:
    column_names = pyarrow.parquet.ParquetFile(parquet_file).schema.names

    # these mappings were recognized so far in the data. If a new schema is encountered, it must be added here.
    possible_schemas = [
        ColumnMapping("tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance"),
        ColumnMapping("Trip_Pickup_DateTime", "Trip_Dropoff_DateTime", "Trip_Distance"),
        ColumnMapping("pickup_datetime", "dropoff_datetime", "trip_distance"),
    ]

    for schema in possible_schemas:
        if schema.matches_columns(column_names):
            return schema

    raise ValueError(
        f"Could not find a matching schema for columns {column_names} in {parquet_file}"
    )


def read_parquet_file_with_unknown_schema(parquet_file: pl.Path) -> pd.DataFrame:
    columns_mapping = get_column_mapping(parquet_file)

    df = pd.read_parquet(parquet_file, columns=columns_mapping.column_names())
    df = df.rename(columns=columns_mapping.mapping_dict())

    return df


def load_parquet_file(month_id: MonthIdentifier) -> pd.DataFrame:
    parquet_file = acquire_parquet_file(month_id)

    df = read_parquet_file_with_unknown_schema(parquet_file)

    if df.isnull().any().any():
        # haven't found Nan values in the data yet, but I want to know if they appear
        raise ValueError("Found NaN values in the loaded DataFrame")

    for time_column in TIME_COLUMNS:
        if time_column not in df.columns:
            raise ValueError(f"Column {time_column} not found in the loaded DataFrame")
        df[time_column] = pd.to_datetime(df[time_column]).dt.tz_localize(
            ASSUMED_ORIGIN_TZ  # sometimes times get not correctly recognized as times. I assume that all times are in UTC
        )

    df["trip_length_time"] = df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]

    return df


def filter_df_for_correct_time(
    df: pd.DataFrame, month_id: MonthIdentifier
) -> pd.DataFrame:
    buffer_slight_overlap = pd.Timedelta(
        1, unit="hour"
    )  # parquet files sometimes include a few entries from the next or previous month

    start_limit = month_id.start_timestamp() - buffer_slight_overlap
    end_limit = month_id.end_timestamp() + buffer_slight_overlap

    out_of_range_indices = (df[DEFINING_TIME_COLUMN] < start_limit) | (
        df[DEFINING_TIME_COLUMN] >= end_limit
    )
    if out_of_range_indices.any():
        LOGGER.warning(
            f"Found {out_of_range_indices.sum()} entries with invalid time in {month_id.year}-{month_id.month:02}. They will be removed."
        )
        for i, row in df[out_of_range_indices].iterrows():
            LOGGER.info(f"Invalid time at index {i}: {row[DEFINING_TIME_COLUMN]}")

    return df[~out_of_range_indices]


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outliers from the DataFrame. This function is very simple and only removes outliers based on a rough estimate.
    It could be improved by using a more sophisticated outlier detection algorithm.
    """
    limits = {
        "trip_distance": (0, 100),
        "trip_length_time": (pd.Timedelta("10s"), pd.Timedelta("24h")),
    }

    for column, (lower, upper) in limits.items():
        outlier_indices = (df[column] < lower) | (df[column] > upper)
        if outlier_indices.any():
            LOGGER.info(
                f"Found {outlier_indices.sum()} outliers in {column}. They will be removed."
            )
        df = df[~outlier_indices]

    return df


def load_filtered_parquet_file(month_id: MonthIdentifier) -> pd.DataFrame:
    df = load_parquet_file(month_id)
    df = filter_df_for_correct_time(df, month_id)
    df = remove_outliers(
        df
    )  # the potential effect of drastic outliers on the data is high, therefore they are removed
    return df


def daily_means_from_df(df: pd.DataFrame) -> pd.DataFrame:
    # here we use the timezone of New York, because the data is from New York
    df["date"] = df["tpep_pickup_datetime"].dt.tz_convert(NEW_YORK_TZ).dt.date

    # also collect count to be able to combine dataframes later
    return df.groupby("date").agg(
        trip_distance=("trip_distance", "mean"),
        trip_length_time=("trip_length_time", "mean"),
        count=("trip_distance", "count"),
    )


def get_daily_means_for_month(month_id: MonthIdentifier) -> pd.DataFrame:
    filtered_df = load_filtered_parquet_file(month_id)
    return daily_means_from_df(filtered_df)


def get_months_in_range_inclusive(
    start: MonthIdentifier, end: MonthIdentifier
) -> List[MonthIdentifier]:
    if start > end:
        raise ValueError("Start month must be before end month")

    months = []
    current = start
    while current <= end:
        months.append(current)
        current = current.next_month()

    return months


def combine_rows_via_weighted_average(row1: pd.Series, row2: pd.Series) -> pd.Series:
    if row1.name != row2.name:
        raise ValueError("Rows must have the same index")

    return pd.Series(
        {
            "trip_distance": row1["trip_distance"] * row1["count"]
            + row2["trip_distance"] * row2["count"],
            "trip_length_time": row1["trip_length_time"] * row1["count"]
            + row2["trip_length_time"] * row2["count"],
            "count": row1["count"] + row2["count"],
        }
    ) / (row1["count"] + row2["count"])


def fix_first_and_last_days_of_consecutive_dfs(
    df_before: pd.DataFrame,
    month_before: MonthIdentifier,
    df_after: pd.DataFrame,
    month_after: MonthIdentifier,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    The assignment of the first and last days of a month to a month is not always exactly correct within the parquet
    files. I.e. the file for april might contain a few samples from 1st of may or 31st of march. In order to not have to
    open multiple large parquet files at once, this function fixes the overlap between two months by combining the
    data of the overlapping days via a weighted average.
    """

    if month_before.next_month() != month_after:
        raise ValueError("Months must be consecutive")

    if df_before.empty or df_after.empty:
        return df_before, df_after

    potentially_overlapping_date = month_after.first_day_of_month()
    if potentially_overlapping_date in df_before.index:
        LOGGER.info(
            f"Fixing {potentially_overlapping_date} that landed in both {month_before} and {month_after}."
        )
        df_after.loc[potentially_overlapping_date] = combine_rows_via_weighted_average(
            df_before.loc[potentially_overlapping_date],
            df_after.loc[potentially_overlapping_date],
        )
        df_before = df_before.drop(potentially_overlapping_date)

    potentially_overlapping_date = month_before.last_day_of_month()
    if potentially_overlapping_date in df_after.index:
        LOGGER.info(
            f"Fixing {potentially_overlapping_date} that landed in both {month_before} and {month_after}."
        )
        df_before.loc[potentially_overlapping_date] = combine_rows_via_weighted_average(
            df_before.loc[potentially_overlapping_date],
            df_after.loc[potentially_overlapping_date],
        )
        df_after = df_after.drop(potentially_overlapping_date)

    return df_before, df_after


def fix_first_and_last_days_in_list_of_dfs(
    month_daily_means_tuples: List[Tuple[MonthIdentifier, pd.DataFrame]],
) -> List[pd.DataFrame]:
    if len(month_daily_means_tuples) == 1:
        return [month_daily_means_tuples[0][1]]

    daily_means = []
    first_month, first_daily_means = month_daily_means_tuples[0]

    for second_month, second_daily_means in month_daily_means_tuples[1:]:
        first_daily_means, second_daily_means = (
            fix_first_and_last_days_of_consecutive_dfs(
                first_daily_means, first_month, second_daily_means, second_month
            )
        )
        daily_means.append(first_daily_means)
        first_month = second_month
        first_daily_means = second_daily_means

    daily_means.append(second_daily_means)
    return daily_means


def get_daily_means_in_range(
    start: MonthIdentifier, end: MonthIdentifier
) -> pd.DataFrame:
    months = get_months_in_range_inclusive(start, end)

    for m in months:
        if not is_url_valid(expected_parquet_file_url(m)):
            raise ValueError(
                f"No data available for {m} is not valid. Consider limiting the range."
            )

    month_daily_means_tuples = [
        (month_id, get_daily_means_for_month(month_id)) for month_id in months
    ]

    if not month_daily_means_tuples:
        raise ValueError("No months found in range")

    daily_means_df_list = fix_first_and_last_days_in_list_of_dfs(
        month_daily_means_tuples
    )
    daily_means_df = pd.concat(daily_means_df_list)
    return daily_means_df