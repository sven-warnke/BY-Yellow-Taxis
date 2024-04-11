import pandas as pd
import pytest

from helpers import parquet_file_handling


class TestMonthIdentifier:
    @pytest.mark.parametrize(
        "year, month",
        [
            (2020, 13),
            (2020, 0),
        ],
    )
    def test_raises_on_invalid_input(self, year, month):
        with pytest.raises(ValueError):
            parquet_file_handling.MonthIdentifier(year, month)

    def test_lt(self):
        assert parquet_file_handling.MonthIdentifier(
            2020, 1
        ) < parquet_file_handling.MonthIdentifier(2020, 2)
        assert not parquet_file_handling.MonthIdentifier(
            2020, 2
        ) < parquet_file_handling.MonthIdentifier(2020, 1)
        assert not parquet_file_handling.MonthIdentifier(
            2020, 1
        ) < parquet_file_handling.MonthIdentifier(2020, 1)

    def test_le(self):
        assert parquet_file_handling.MonthIdentifier(
            2020, 1
        ) <= parquet_file_handling.MonthIdentifier(2020, 2)
        assert not parquet_file_handling.MonthIdentifier(
            2020, 2
        ) <= parquet_file_handling.MonthIdentifier(2020, 1)
        assert parquet_file_handling.MonthIdentifier(
            2020, 1
        ) <= parquet_file_handling.MonthIdentifier(2020, 1)

    def test_eq(self):
        assert parquet_file_handling.MonthIdentifier(
            2020, 1
        ) == parquet_file_handling.MonthIdentifier(2020, 1)
        assert not parquet_file_handling.MonthIdentifier(
            2020, 1
        ) == parquet_file_handling.MonthIdentifier(2020, 2)

    @pytest.mark.parametrize(
        "year, month, expected",
        [
            (2020, 1, pd.Timestamp("2020-01", tz="UTC")),
            (2020, 12, pd.Timestamp("2020-12", tz="UTC")),
        ],
    )
    def test_start_timestamp(self, year, month, expected):
        month_id = parquet_file_handling.MonthIdentifier(year, month)
        assert month_id.start_timestamp() == expected

    @pytest.mark.parametrize(
        "year, month, exclusive_end",
        [
            (2020, 1, pd.Timestamp("2020-02-01", tz="UTC")),
            (2020, 12, pd.Timestamp("2021-01-01", tz="UTC")),
        ],
    )
    def test_end_timestamp(self, year, month, exclusive_end):
        month_id = parquet_file_handling.MonthIdentifier(year, month)
        assert month_id.end_timestamp() < exclusive_end
        assert month_id.end_timestamp() - exclusive_end < pd.Timedelta("1ms")


def test_combine_dfs_via_weighted_average():
    df1 = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "trip_distance": [1, 2, 3],
            "trip_length_time": [
                pd.Timedelta("1 hour"),
                pd.Timedelta("2 hours"),
                pd.Timedelta("3 hours"),
            ],
            "count": [1, 1, 1],
        }
    )
    df1 = df1.set_index("date")
    df2 = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "trip_distance": [2, 3, 4],
            "trip_length_time": [
                pd.Timedelta("2 hours"),
                pd.Timedelta("3 hours"),
                pd.Timedelta("4 hours"),
            ],
            "count": [2, 2, 2],
        }
    )
    df2 = df2.set_index("date")
    result = parquet_file_handling.combine_dfs_via_weighted_average(df1, df2)
    expected = pd.DataFrame(
        {
            "trip_distance": [5 / 3, 8 / 3, 11 / 3],
            "trip_length_time": [
                pd.Timedelta(5 / 3, "hours"),
                pd.Timedelta(8 / 3, "hours"),
                pd.Timedelta(11 / 3, "hours"),
            ],
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "count": [3, 3, 3],
        }
    )
    expected = expected.set_index("date")
    expected["trip_length_time"] = expected[
        "trip_length_time"
    ].dt.total_seconds()  # convert to seconds because assert_frame_equal does not support Timedelta
    result["trip_length_time"] = result["trip_length_time"].dt.total_seconds()
    pd.testing.assert_frame_equal(result, expected, check_exact=False)
