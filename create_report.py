import argparse
import pathlib as pl

from helpers import parquet_file_handling, plotting, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2009, help="Start year")
    parser.add_argument("--start-month", type=int, default=1, help="Start month")
    parser.add_argument(
        "--end-year", type=int, default=2024, help="End year (inclusive)"
    )
    parser.add_argument(
        "--end-month", type=int, default=1, help="End month (inclusive)"
    )
    parser.add_argument(
        "--output-path",
        type=pl.Path,
        default=pl.Path("reports/plots.html"),
        help="Output path of the report",
    )
    args = parser.parse_args()

    start_month_id = parquet_file_handling.MonthIdentifier(
        args.start_year, args.start_month
    )
    end_month_id = parquet_file_handling.MonthIdentifier(args.end_year, args.end_month)
    daily_means_df = parquet_file_handling.get_daily_means_in_range(
        start_month_id, end_month_id
    )

    monthly_mean_fig = plotting.plot_rolling_means_for_time_and_distance(
        daily_means_df, plotting.time_wise_averaging.MonthlyMean()
    )

    rolling_mean_fig = plotting.plot_rolling_means_for_time_and_distance(
        daily_means_df, plotting.time_wise_averaging.RollingMean()
    )

    report.save_plots(
        figures=[monthly_mean_fig, rolling_mean_fig],
        filename=args.output_path,
    )


if __name__ == "__main__":
    main()
