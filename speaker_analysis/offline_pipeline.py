import typing as t
import pandas as pd

from models.asr import SpeechRecognizer
from models.diarization import Diarizer
from models.sentiment import TranscriptSentimentizer
from models.stress_detection import StressDetector


class OfflineSpeechProcessingPipeline:
    def __init__(self, hf_access_token: str):
        # Load models that are independent of speaker
        self.sr = SpeechRecognizer()
        self.diarizer = Diarizer(access_token=hf_access_token)
        self.sentimentizer = TranscriptSentimentizer()

    def _parse(self, file_path: str) -> t.List[dict]:
        """Applies the diarizer to segment audio by speaker and time."""
        return self.diarizer.apply(file_path=file_path)

    def analyze(self, file_path: str, stress_sensitivity: float) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
        """Applies speech analysis to the provided file."""
        # First diarize to segment audio by speaker
        parsed = self._parse(file_path=file_path)

        # Add properties to each parsed segment
        # TODO: how to batch this?
        parsed_cumulative = []
        for segment in parsed:
            # Perform ASR
            transcript = self.sr.transcribe_audio(byte_array=segment['bytes'])
            segment.update({"transcript": transcript})

            # Detect stress in speech - each speaker should have their own stress detection model
            segment.update({"stress_detector": StressDetector()})
            stress = segment["stress_detector"].detect(audio=segment['bytes'], sensitivity=stress_sensitivity)
            segment.update({"is_stressed": stress})

            # No longer need these
            del segment["stress_detector"]
            del segment["bytes"]

            # Analyze sentiment
            sentiment = self.sentimentizer.apply(text=transcript)
            segment.update({"sentiment": sentiment})

            # Append 2 new events to the cumulative version of parsed segments
            parsed_cumulative.append(
                {
                    "timestamp": segment['timestamp_start'],
                    "speaker": segment['speaker'],
                    "color": segment['color'],
                    "duration": 0.0
                }
            )
            parsed_cumulative.append(
                {
                    "timestamp": segment['timestamp_end'],
                    "speaker": segment['speaker'],
                    "color": segment['color'],
                    "duration": segment['duration']
                }
            )

        # Convert each parsed segment to a row in a dataframe, for plotting with Plotly/Dash
        df = pd.DataFrame(parsed)

        # Do the same for the cumulative version
        dfc = pd.DataFrame(parsed_cumulative)
        dfc['cumulative_speaking_time'] = dfc.groupby('speaker')['duration'].transform('cumsum')

        return df, dfc
