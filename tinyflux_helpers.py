import pathlib as pl
from typing import List

import pandas as pd
import tinyflux


def get_tinyflux_db(data_folder: pl.Path) -> tinyflux.TinyFlux:
    data_folder.mkdir(parents=True, exist_ok=True)
    return tinyflux.TinyFlux(data_folder / 'tinyflux_db.csv')


def insert_df_into_tinyflux(
        df: pd.DataFrame,
        tinyflux_db: tinyflux.TinyFlux,
        time_column: str,
        field_columns: List[str]
) -> None:
    for index, row in df.iterrows():
        time = row[time_column]
        fields = {col: row[col] for col in field_columns}
        point = tinyflux.Point(time=time, fields=fields)

        tinyflux_db.insert(point)
