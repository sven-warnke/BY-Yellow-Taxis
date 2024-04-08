import parquet_file_handling
import pytest



def test_month_identifier():
    with pytest.raises(ValueError):
        parquet_file_handling.MonthIdentifier(2020, 13)
    with pytest.raises(ValueError):
        parquet_file_handling.MonthIdentifier(2020, 0)
    assert parquet_file_handling.MonthIdentifier(2020, 1) < parquet_file_handling.MonthIdentifier(2020, 2)
    assert parquet_file_handling.MonthIdentifier(2020, 1) <= parquet_file_handling.MonthIdentifier(2020, 2)
    assert parquet_file_handling.MonthIdentifier(2020, 1) == parquet_file_handling.MonthIdentifier(2020, 1)
    assert parquet_file_handling.MonthIdentifier(2020, 2) > parquet_file_handling.MonthIdentifier(2020, 1)

