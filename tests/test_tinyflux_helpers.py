import tinyflux_helpers
import pathlib as pl
import tempfile


def test_get_tinyflux_db_works_on_non_existing_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tinyflux_helpers.get_tinyflux_db(pl.Path(tmp_dir) / "bla")

