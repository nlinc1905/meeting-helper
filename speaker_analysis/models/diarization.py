import torch
import typing as t
from pyannote.audio import Pipeline
from pydub import AudioSegment

from models.utils import assign_color_to_category


class Diarizer:
    def __init__(self, access_token: str):
        # Hardcode this model until something better comes along
        model_name = "pyannote/speaker-diarization-3.1"

        # Set up a pipeline for speaker diarization and send to GPU if available
        self.pipeline = Pipeline.from_pretrained(model_name, use_auth_token=access_token)
        if torch.cuda.is_available():
            self.pipeline.to(torch.device("cuda"))

    @staticmethod
    def slice_audio(audio_segment: AudioSegment, start: float, end: float) -> bytearray:
        """
        Slices a Pydub AudioSegment object at the given start and end points.

        :param audio_segment: Pydub AudioSegment object
        :param start: Time to start at, expressed in seconds, e.g. 0.1234
        :param end: Time to end at, expressed in seconds, e.g. 0.1234
        """
        # Split the audio, multiply by 1000 to do it by millisecond
        byte_array = audio_segment[start * 1_000:end * 1_000].get_array_of_samples()
        return byte_array

    def apply(self, file_path: str) -> t.List[dict]:
        """Apply the pretrained pipeline for speaker diarization"""
        diarization = self.pipeline(file_path)
        audio = AudioSegment.from_wav(file_path)

        # Convert events to a data structure
        speak_events = [
            {
                "timestamp_start": turn.start,
                "timestamp_end": turn.end,
                "duration": turn.end - turn.start,
                "speaker": speaker,
                "bytes": self.slice_audio(audio_segment=audio, start=turn.start, end=turn.end)
            } for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

        # Assign colors to speakers for plotting
        unique_speakers = set(diarization.labels())
        speaker_colors = assign_color_to_category(categories=unique_speakers)
        for d in speak_events:
            d.update({"color": speaker_colors[d["speaker"]]})

        return speak_events
