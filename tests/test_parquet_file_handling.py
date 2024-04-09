import parquet_file_handling
import pytest
import pandas as pd


def test_month_identifier():
    with pytest.raises(ValueError):
        parquet_file_handling.MonthIdentifier(2020, 13)
    with pytest.raises(ValueError):
        parquet_file_handling.MonthIdentifier(2020, 0)
    assert parquet_file_handling.MonthIdentifier(2020, 1) < parquet_file_handling.MonthIdentifier(2020, 2)
    assert parquet_file_handling.MonthIdentifier(2020, 1) <= parquet_file_handling.MonthIdentifier(2020, 2)
    assert parquet_file_handling.MonthIdentifier(2020, 1) == parquet_file_handling.MonthIdentifier(2020, 1)
    assert parquet_file_handling.MonthIdentifier(2020, 2) > parquet_file_handling.MonthIdentifier(2020, 1)


class TestMonthIdentifier:
    @pytest.mark.parametrize('year, month, expected', [
        (2020, 1, pd.Timestamp('2020-01', tz='UTC')),
        (2020, 12, pd.Timestamp('2020-12', tz='UTC')),
    ])
    def test_start_timestamp(self, year, month, expected):
        month_id = parquet_file_handling.MonthIdentifier(year, month)
        assert month_id.start_timestamp() == expected

    @pytest.mark.parametrize('year, month, exclusive_end', [
        (2020, 1, pd.Timestamp('2020-02-01', tz='UTC')),
        (2020, 12, pd.Timestamp('2021-01-01', tz='UTC')),
    ])
    def test_end_timestamp(self, year, month, exclusive_end):
        month_id = parquet_file_handling.MonthIdentifier(year, month)
        assert month_id.end_timestamp() < exclusive_end
        assert month_id.end_timestamp() - exclusive_end < pd.Timedelta('1ms')
