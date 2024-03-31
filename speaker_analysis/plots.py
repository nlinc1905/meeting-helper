import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def make_stress_plot(df: pd.DataFrame):
    fig = make_subplots()

    # Add a dummy trace - the rectangles and legend won't show up if a trace hasn't been added first
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', line=dict(color='white', width=0), showlegend=False))

    # # Fill the area above 0 with red (high stress)
    # fig.add_hrect(
    #     y0=0,
    #     y1=1,
    #     fillcolor="red",
    #     opacity=0.15,
    #     line_width=0,
    # )
    #
    # # Fill the area below 0 with green (low stress)
    # fig.add_hrect(
    #     y0=0,
    #     y1=-1,
    #     fillcolor="green",
    #     opacity=0.15,
    #     line_width=0,
    # )

    # Add a bar plot trace for each speaker, where bars are colored by speaker,
    # the width of each bar = speaking duration, and the direction of the bar
    # in relation to 0 corresponds with stress (+1) or no stress (-1).
    for p in set(df['speaker']):
        df_sub = df[df['speaker'] == p]
        fig.add_trace(
            go.Bar(
                name=p,
                x=df_sub['timestamp_start'] + (df_sub['duration'] / 2),  # center on the midpoint of the segment
                y=df_sub['is_stressed'],
                width=df_sub['duration'],
                marker_color=df_sub['color'],
                customdata=np.stack([df_sub['speaker'], df_sub['transcript']], axis=1),
                hovertemplate='%{customdata[1]}'
            ),
        )

    # Finalize layout
    fig.update_yaxes(range=[-1, 1])
    fig.update_layout(
        title=dict(text="Comments and Stress Over Time", font=dict(size=30)),
        xaxis_title="Time",
        yaxis_title="Stress",
        legend_title="Speaker",
        autosize=False,
        width=850,
        height=300,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=50,
            pad=0
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=[-1, 0, 1],
            ticktext=[
                "<span style='color:green; font-weight:bold'>No Stress</span>",
                '----------',
                "<span style='color:red; font-weight:bold'>Stress</span>"
            ]
        ),
        # paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig


def make_cumulative_stress_plot(df: pd.DataFrame):
    fig = make_subplots()

    # Add a trace for the cumulative stress line
    fig.add_trace(
        go.Scatter(
            name="Aggregate",
            x=df['timestamp_end'],
            y=df['cumulative_stress'],
            mode='lines',
            line=dict(color='black', width=1),
        )
    )

    # Fill the area above 0 with red (high stress)
    fig.add_hrect(
        y0=0,
        y1=max(10, df['cumulative_stress'].max() + 1),
        fillcolor="red",
        opacity=0.15,
        line_width=0,
    )

    # Fill the area below 0 with green (low stress)
    fig.add_hrect(
        y0=0,
        y1=min(-10, df['cumulative_stress'].min() - 1),
        fillcolor="green",
        opacity=0.15,
        line_width=0,
    )

    # Add a scatterplot trace for each speaker, where points are colored by speaker
    for p in set(df['speaker']):
        df_sub = df[df['speaker'] == p]
        fig.add_trace(
            go.Scatter(
                name=p,
                x=df_sub['timestamp_end'],
                y=df_sub['cumulative_stress'],
                mode='markers',
                marker_color=df_sub['color'],
                marker_size=12,
                customdata=np.stack([df_sub['speaker'], df_sub['transcript']], axis=1),
                hovertemplate='%{customdata[1]}'
            ),
        )

    # Finalize layout
    fig.update_layout(
        title=dict(text="Comments and Stress Over Time", font=dict(size=30)),
        xaxis_title="Time",
        yaxis_title="Cumulative Comment Stress",
        legend_title="Speaker",
        autosize=False,
        width=850,
        height=400,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=50,
            pad=0
        ),
    )

    return fig


def make_cumulative_speaking_time_chart(df: pd.DataFrame):
    # Make a line chart showing cumulative speaking time, colored by speaker
    # cumulative_speaking_time_fig = go.Figure(
    #     data=go.Scatter(
    #         x=df["timestamp"],
    #         y=df["cumulative_speaking_time"],
    #         marker=dict(color=df["color"], size=12),
    #         mode="lines+markers",
    #         customdata=np.stack([df['speaker']], axis=1),
    #         hovertemplate='%{customdata[1]}'
    #     )
    # )
    cumulative_speaking_time_fig = px.line(
        df,
        x="timestamp",
        y="cumulative_speaking_time",
        color="speaker",
        color_discrete_map=dict(zip(df['speaker'], df['color'])),
        markers=True,
    )

    # Finalize the layout
    cumulative_speaking_time_fig.update_traces(marker=dict(size=12))
    cumulative_speaking_time_fig.update_xaxes(range=[0, max(df['timestamp'])])
    cumulative_speaking_time_fig.update_yaxes(range=[0, max(df['cumulative_speaking_time']) + 1.0])
    cumulative_speaking_time_fig.update_layout(
        title=dict(text="Cumulative Speaking Time by Speaker", font=dict(size=30)),
        xaxis_title="Time",
        yaxis_title="Cumulative Speaking Time (seconds)",
        legend_title="Speaker",
        autosize=False,
        width=850,
        height=400,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=50,
            pad=0
        ),
    )

    return cumulative_speaking_time_fig


def make_speech_frequency_bar_chart(df: pd.DataFrame):
    # Ignore any times duration is 0
    df = df[df['duration'] != 0.0]

    # Group the data by speaker and count the number of times they spoke
    speech_freq = df.groupby('speaker')['timestamp_start'].count().reset_index().rename(
        columns={"timestamp_start": "nbr_times_spoken"}
    )

    # Make a bar chart for the number of times each speaker spoke
    speech_freq_fig = px.bar(
        speech_freq,
        x='speaker',
        y='nbr_times_spoken',
        title="Times Spoken",
        color="speaker",
        color_discrete_map=dict(zip(df['speaker'], df['color'])),
    )

    # Finalize the layout
    speech_freq_fig.update_layout(
        title=dict(text="Total Times Each Speaker Spoke", font=dict(size=30)),
        xaxis_title="Speaker",
        yaxis_title="Nbr Times Spoke",
        legend_title="Speaker",
        autosize=False,
        width=850,
        height=400,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=50,
            pad=0
        ),
    )

    return speech_freq_fig
