import pandas as pd
from helpers import time_wise_averaging


def get_df():
    df = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "trip_distance": [1, 2, 3],
            "trip_length_time": [pd.Timedelta("1 hour"), pd.Timedelta("2 hours"), pd.Timedelta("3 hours")],
        }
    )
    df = df.set_index("date")
    return df


class TestRollingMean:
    
    def test_calculate_mean(self):
        df = get_df()
        rolling_mean = time_wise_averaging.RollingMean()
        result = rolling_mean.calculate_mean(df)
        expected = pd.DataFrame(
            {
                "trip_distance": [1, 1.5, 2],
                "trip_length_in_mins": [60., 90., 120.],
                "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            }
        )
        pd.testing.assert_frame_equal(result, expected)


class TestMonthlyMean:
        
        def test_calculate_mean(self):
            df = get_df()
            monthly_mean = time_wise_averaging.MonthlyMean()
            result = monthly_mean.calculate_mean(df)
            expected = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2020-01-31"]),
                    "trip_distance": [2.],
                    "trip_length_in_mins": [120.],
                }
            )
            pd.testing.assert_frame_equal(result, expected)