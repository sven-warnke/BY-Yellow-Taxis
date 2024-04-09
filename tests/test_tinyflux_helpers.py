import pathlib as pl
import tempfile

import tinyflux_helpers
import parquet_file_handling


def test_get_tinyflux_db_works_on_non_existing_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tinyflux_helpers.get_tinyflux_db(pl.Path(tmp_dir) / "bla")


def test_insert_df_into_tinyflux_works():
    tinyflux_db = tinyflux_helpers.get_tinyflux_db(pl.Path(tempfile.gettempdir()))
    df = parquet_file_handling.load_filtered_parquet_file(parquet_file_handling.MonthIdentifier(2023, 1))

    tinyflux_helpers.insert_df_into_tinyflux(df, tinyflux_db, parquet_file_handling.DEFINING_TIME_COLUMN, ['trip_distance'])
    assert len(tinyflux_db) == len(df)
    assert tinyflux_db.get_all_points() == [tinyflux_helpers.point_from_row(row) for _, row in df.iterrows()]
    assert tinyflux_db.get_all_points() == [tinyflux_helpers.point_from_row(row) for _, row in df.iterrows()]
    assert tinyflux_db.get_all_points() == [tinyflux_helpers.point_from_row(row) for _, row in df.iterrows()]

