import argparse

from helpers import parquet_file_handling, plotting


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2009)
    parser.add_argument("--start-month", type=int, default=1)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--end-month", type=int, default=1)
    args = parser.parse_args()

    start_month_id = parquet_file_handling.MonthIdentifier(
        args.start_year, args.start_month
    )
    end_month_id = parquet_file_handling.MonthIdentifier(args.end_year, args.end_month)
    daily_means_df = parquet_file_handling.get_daily_means_in_range(
        start_month_id, end_month_id
    )
    monthly_mean = plotting.time_wise_averaging.MonthlyMean()
    monthly_mean_fig = plotting.plot_rolling_means_for_time_and_distance(
        daily_means_df, monthly_mean
    )
