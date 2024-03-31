import torch
import pyaudio
import numpy as np
from faster_whisper import WhisperModel

from models.utils import audio_to_numpy


class SpeechRecognizer:
    def __init__(self):
        # It is best to leave sample rate (samples_per_second) alone, as Whisper was trained on a sample rate of 16,000
        self.params = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "nbr_bits_per_sample": pyaudio.paInt16,
            "channels": 1,
            "samples_per_second": 16_000,
            "record_in_chunks_of_this_many_samples": 2_048,
        }
        self.model = WhisperModel("small.en", device=self.params['device'], compute_type="float32")

    def _decode_with_whisper(self, x: np.ndarray, conditional_prompt: str = None) -> str:
        """Decodes an audio array with Whisper."""
        condition_on_prompt = True if conditional_prompt else False
        segments, info = self.model.transcribe(
            x,
            beam_size=5,
            initial_prompt=conditional_prompt,
            condition_on_previous_text=condition_on_prompt
        )
        whisper_text = ''.join([s.text for s in segments])
        return whisper_text

    def transcribe_audio(self, byte_array: bytearray, conditional_prompt: str = None) -> str:
        """
        Use Whisper to transcribe an audio segment.

        :param byte_array: Bytes from audio
        :param conditional_prompt: Text that comes before the audio to be transcribed to assist
            with transcription.
        """
        x = audio_to_numpy(audio_buffer=byte_array)
        text = self._decode_with_whisper(x=x, conditional_prompt=conditional_prompt)
        return text.strip()
