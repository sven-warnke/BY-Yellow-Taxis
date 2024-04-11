import pandas as pd
from plotly import express as px
from plotly import graph_objects as go
from plotly import subplots


def plot_metric(df: pd.DataFrame, metric: str) -> go.Figure:
    fig = px.line(df, x="date", y=metric, markers=True)
    return fig


def plot_rolling_means_for_time_and_distance(
    daily_means_df: pd.DataFrame, time_wise_averager: TimeWiseAverager
) -> go.Figure:
    averaged_df = time_wise_averager.calculate_mean(daily_means_df)
    distance_subplot = plot_metric(averaged_df, "trip_distance")
    time_subplot = plot_metric(averaged_df, "trip_length_in_mins")

    # the speed is not part of the problem statement, but it is a simple calculation that can be added
    # and it is a useful metric for the data to draw conclusions from
    averaged_df["speed_in_mph"] = averaged_df["trip_distance"] / (
        averaged_df["trip_length_in_mins"] / 60
    )

    speed_subplot = plot_metric(averaged_df, "speed_in_mph")

    fig = subplots.make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Trip distance in miles",
            "Trip length in minutes",
            "Speed in mph",
        ),
    )
    fig.add_trace(
        distance_subplot.data[0],
        row=1,
        col=1,
    )
    fig.add_trace(
        time_subplot.data[0],
        row=2,
        col=1,
    )
    fig.add_trace(
        speed_subplot.data[0],
        row=3,
        col=1,
    )

    return fig
