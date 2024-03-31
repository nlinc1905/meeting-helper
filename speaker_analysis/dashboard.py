import argparse
from dash import Dash, dcc, html

from offline_pipeline import OfflineSpeechProcessingPipeline
from plots import make_stress_plot, make_cumulative_speaking_time_chart, make_speech_frequency_bar_chart


app = Dash(__name__)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='A dashboard that displays speaker analysis from a meeting recording.'
    )
    parser.add_argument(
        '-hat',
        '--huggingface_access_token',
        help=(
            'Your Huggingface access token for using the pyannote speaker diarization model. '
            'You can create one from your HF account here: https://huggingface.co/settings/tokens'
            '\nPlease make sure you have also accepted the user agreement on HF model hub for pyannote:'
            'https://huggingface.co/pyannote/speaker-diarization-3.1'
        ),
        type=str,
    )
    parser.add_argument(
        '-ss',
        '--stress_sensitivity',
        help=(
            "A float between 0.0 and 1.0 that indicates which percentage higher each successive audio"
            "buffer's F0 statistics need to be above the max of the previous values, in order to be"
            "considered anomalous and indicate that the speaker is stressed."
        ),
        nargs='?',
        default=0.1,
        type=float,
    )
    args = vars(parser.parse_args())

    # Analyze the meeting recording and prepare data for the dashboard
    ospp = OfflineSpeechProcessingPipeline(hf_access_token=args['huggingface_access_token'])
    df, dfc = ospp.analyze(file_path="data/meeting_recording.wav", stress_sensitivity=args['stress_sensitivity'])

    app.title = "Meeting Analysis"
    app.layout = html.Div(
        children=[
            html.Div(
                id="plot_container",
                children=[
                    dcc.Graph(id="aggregate_stress", figure=make_stress_plot(df=df)),
                    html.Br(),
                    dcc.Graph(id="cumulative_speaking_times", figure=make_cumulative_speaking_time_chart(df=dfc)),
                    html.Br(),
                    dcc.Graph(id="speech_freq", figure=make_speech_frequency_bar_chart(df=df))
                ],
                style={"border": "2px solid gray", "margin": "0.5em", "padding": "2em"}
            )
        ]
    )

    app.run_server(debug=False, use_reloader=False)
