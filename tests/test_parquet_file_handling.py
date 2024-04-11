from helpers import parquet_file_handling
import pytest
import pandas as pd


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
