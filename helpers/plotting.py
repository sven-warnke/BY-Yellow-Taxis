import pandas as pd
from plotly import express as px
from plotly import graph_objects as go
from plotly import subplots

from helpers import time_wise_averaging
from helpers import constants as c


def plot_metric_by_date(df: pd.DataFrame, metric: str) -> go.Figure:
    fig = px.line(df, x=c.DATE_COLUMN, y=metric, markers=True)
    return fig


def plot_rolling_means_for_time_and_distance(
    daily_means_df: pd.DataFrame,
    time_wise_averager: time_wise_averaging.TimeWiseAverager,
) -> go.Figure:
    averaged_df = time_wise_averager.calculate_mean(daily_means_df)
    distance_subplot = plot_metric_by_date(averaged_df, c.DISTANCE_COLUMN)
    time_subplot = plot_metric_by_date(averaged_df, c.LENGTH_IN_MINS_COLUMN)

    # the speed is not part of the problem statement, but it is a simple calculation that can be added
    # and it is a useful metric for the data to draw conclusions from
    averaged_df[c.SPEED_COLUMN] = averaged_df[c.DISTANCE_COLUMN] / (
        averaged_df[c.LENGTH_IN_MINS_COLUMN] / 60
    )

    speed_subplot = plot_metric_by_date(averaged_df, c.SPEED_COLUMN)

    fig = subplots.make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Trip distance in miles",
            "Trip length in minutes",
            "Speed in mph",
        )        
    )
    fig.update_layout(title_text=time_wise_averager.name())
    
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
