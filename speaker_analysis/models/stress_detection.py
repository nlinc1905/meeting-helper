import librosa
import typing as t
import numpy as np
from collections import deque

from models.utils import audio_to_numpy


class StressDetector:
    def __init__(self, sample_rate: int = 8_000, nbr_buffers_to_queue: int = 3):
        """
        Sets up a stress detector.  The detector is unique to an individual speaker.  Every speaker
        has different characteristics and should have their own StressDetector instance.

        :param sample_rate: Audio sample rate in Hz
        :param nbr_buffers_to_queue: How many audio buffers to include in the rolling statistics.
        """
        self.sample_rate = sample_rate
        self.mono = True  # only mono is supported
        self.baseline_nbr_samples = nbr_buffers_to_queue
        self.means = deque([], maxlen=nbr_buffers_to_queue)
        self.stds = deque([], maxlen=nbr_buffers_to_queue)
        self.mins = deque([], maxlen=nbr_buffers_to_queue)
        self.maxs = deque([], maxlen=nbr_buffers_to_queue)

    def detect(self, audio: t.Union[bytearray, str], n_thresholds: int = 2, sensitivity: float = 0.3) -> int:
        """
        Detects stress by relying on rolling statistics of the fundamental frequency, F0.
        Although F0 is commonly used in vocal stress detection, this method has not been evaluated
        on labeled datasets, so it is likely not good.

        The F0 statistics are calculated on an entire audio buffer, meaning it will be impossible
        to detect stress within a buffer.  Anomalies will therefore refer to entire buffers that differ
        significantly from the buffers that came before it.  The implication is that stress does not
        surface while a speaker is speaking, but as the result of an interaction with other speakers.
        So if the speaker is distressed by what was just said by another person, that should impact the
        entirety of the speaker's audio buffer, and it will cause this function to return True.

        :param audio: bytearray or the string name of a file to process
        :param n_thresholds: The number of thresholds to use when sampling from the prior (beta) distribution of YIN
            thresholds.  The original paper used 100 (the default in Librosa), resulting in thresholds from
            0.01 to 1, with increments of 1/100.  The smaller this value, the smoother the f0 graph (fewer
            NaNs), but more frequency changes will be considered voiced.   The larger this value, the more
            fragmented the f0 graph (more NaNs), as fewer frequency changes will be considered voiced.
            The parameter degrades quickly, with n_thresholds=3 being similar to n_thresholds=100.  This value
            should be set depending on what the fundamental frequency is being measured for.  If it is audio
            with speech, F0 should come from the voiced segments, suggesting a larger n_thresholds would be
            better.  If it is background noise, then n_thresholds=1 might be better, because the noise is constant.
        :param sensitivity: A float between 0.0 and 1.0 that indicates which percentage higher the current audio
            buffer's F0 statistics need to be above the max of the previous values in the queue, in order to be
            considered anomalous.
        """
        if isinstance(audio, str):
            # Load from file
            wav, _ = librosa.load(audio, sr=self.sample_rate, mono=self.mono)
        else:
            # Convert bytearray to numpy float 32
            wav = audio_to_numpy(audio_buffer=audio)

        # Calculate the F0 for the audio buffer
        # f0 = librosa.yin(wav, fmin=librosa.note_to_hz('C0'), fmax=librosa.note_to_hz('C7'))
        f0, voiced_flag, voiced_prob = librosa.pyin(
            wav,
            sr=self.sample_rate,
            fmin=librosa.note_to_hz('C0'),
            fmax=librosa.note_to_hz('C7'),
            frame_length=2_048,  # nbr of samples in a frame
            n_thresholds=n_thresholds,
        )

        # Get F0 statistics
        f0_mean = np.nanmean(f0)
        f0_std = np.nanstd(f0)
        f0_min = np.nanmin(f0)
        f0_max = np.nanmax(f0)

        # Check if at least 2 of the statistics are >= 30% higher than the max of their queue.
        # If they are, then consider the current audio buffer to be anomalous.
        anomaly_detected = False
        if len(self.means) == self.baseline_nbr_samples:
            # Convert each boolean to an integer and sum them
            total = (
                int(f0_mean >= (1 + sensitivity) * np.max(self.means))
                + int(f0_std >= (1 + sensitivity) * np.max(self.stds))
                + int(f0_min >= (1 + sensitivity) * np.max(self.mins))
                + int(f0_max >= (1 + sensitivity) * np.max(self.maxs))
            )
            if total >= 2:
                anomaly_detected = True

        # Update rolling values, only if the update is a "normal" value
        if not anomaly_detected:
            self.means.append(f0_mean)
            self.stds.append(f0_std)
            self.mins.append(f0_min)
            self.maxs.append(f0_max)

        # Convert from boolean to +1 for stress or -1 no stress
        anomaly_detected = -1 if int(anomaly_detected) == 0 else int(anomaly_detected)

        return anomaly_detected
