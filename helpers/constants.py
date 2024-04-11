import pathlib as pl

DATA_FOLDER = pl.Path(__file__).parent.parent / "data"
INTERMEDIATE_DATA_FOLDER = DATA_FOLDER / "intermediate"

PICKUP_TIME_COLUMN = "pickup_datetime"
DROPOFF_TIME_COLUMN = "dropoff_datetime"
DISTANCE_COLUMN = "trip_distance"
TIME_COLUMNS = [PICKUP_TIME_COLUMN, DROPOFF_TIME_COLUMN]
DEFINING_TIME_COLUMN = PICKUP_TIME_COLUMN

# columns created by the program
TIME_LENGTH_COLUMN = "trip_length_time"
DATE_COLUMN = "date"
COUNT_COLUMN = "count"
LENGTH_IN_MINS_COLUMN = "trip_length_in_mins"
SPEED_COLUMN = "speed_in_mph"

ASSUMED_ORIGIN_TZ = "UTC"
NEW_YORK_TZ = "US/Eastern"
