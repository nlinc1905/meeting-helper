import argparse
import pyaudio
import torch
import numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel
from transformers import pipeline
from threading import Thread
from pynput.keyboard import KeyCode, Key, Listener
from colorama import init, Back, Style

from speaker_analysis.models.utils import audio_to_numpy


def on_press(key: KeyCode) -> None:
    """What to do when a key is pressed."""
    pass


def on_release(key: KeyCode) -> bool:
    """What to do when a key is released."""
    if key == Key.esc:
        # Return False to stop the listener's thread
        return False
    else:
        return True


def decode_with_whisper(model: WhisperModel, x: np.ndarray, conditional_prompt: str = None) -> str:
    """Decodes an audio tensor with Whisper."""
    condition_on_prompt = True if conditional_prompt else False
    segments, info = model.transcribe(
        x,
        beam_size=5,
        initial_prompt=conditional_prompt,
        condition_on_previous_text=condition_on_prompt
    )
    whisper_text = ''.join([s.text for s in segments])
    return whisper_text


def print_summary(text_to_summarize: str) -> None:
    summary = summarizer(text_to_summarize)
    print(f"\n--------\nSummary of the last 60 seconds:\n{summary[0]['summary_text']}\n--------\n")


def save_recording_to_wav(recording: list, sample_rate: int, file_path: str) -> None:
    """
    Converts raw recording to a numpy 16 bit array and writes to a .wav file.

    :param recording: List of bytearrays
    :param sample_rate: Samples per second
    :param file_path: Where to save .wav file
    """
    np16array = [np.frombuffer(audio_buffer, dtype=np.int16) for audio_buffer in recording]
    wavfile.write(file_path, sample_rate, np.hstack(np16array))


def record(
    buffer_rate: int,
    sample_chunks: int,
    buffer_len_seconds: int,
    nbr_past_buffers_to_summarize: int,
    recording: list,
    buffers: list,
    texts: list,
    word_to_highlight: str,
) -> None:
    """
    Records audio and calls the models for ASR and summarization.

    :param buffer_rate: Same as the sample rate - how many samples per second.
    :param sample_chunks: How many discrete chunks to break the audio buffer into.
    :param buffer_len_seconds: How many seconds to include in a buffer.
    :param nbr_past_buffers_to_summarize: How many buffers to include in the rolling summary.
    :param recording: A list of bytearrays of the raw recorded audio stream.
    :param buffers: A list of previously transcribed audio buffers as numpy float32 arrays.  It starts as an empty list.
    :param texts: A list of previously transcribed text strings.  It starts as an empty list.
    :param word_to_highlight: This word is highlighted when the transcribed text is printed to the terminal.
    """
    # Whisper is trained on 30 second clips, so we need to know when enough chunks are available for ASR
    checkpoint = int(buffer_rate * buffer_len_seconds / sample_chunks)
    # We also need to know when enough buffers have been transcribed to summarize
    checkpoint_summarize = checkpoint * nbr_past_buffers_to_summarize
    i, j = 0, 0  # to track progress towards next ASR transcription and summarization, respectively

    while True:

        try:
            # record an audio buffer as a bytearray,
            # transform it to a numpy array,
            # and append it to the list of all buffers
            data = stream.read(sample_chunks)
            recording.append(data)
            buffers.append(audio_to_numpy(audio_buffer=data))

            if i == checkpoint:
                # combine the buffer arrays that have been recorded so far
                x = np.hstack(buffers)

                # perform ASR on buffers recorded so far
                text = decode_with_whisper(
                    model=model,
                    x=x,
                    conditional_prompt=(texts[-1] if len(texts) > 0 else None)
                )

                # save newly transcribed text to ongoing record
                texts.append(text)

                # print while highlighting important word
                if word_to_highlight and word_to_highlight.lower() in text.lower():
                    i = text.lower().index(word_to_highlight.lower())
                    print(text[:i] + Style.BRIGHT + Back.YELLOW + word_to_highlight, text[i+4:])
                else:
                    print(text)

                # clear out the buffers
                buffers = []
                i = 0

            if j == checkpoint_summarize:
                # summarize the last n buffers
                text_to_summarize = " ".join(texts[-nbr_past_buffers_to_summarize:])
                Thread(target=print_summary, args=(text_to_summarize,), name="print_summary").start()

                # reset counter
                j = 0

            # print('Time to next transcription:', checkpoint - i)
            i += 1
            j += 1

        # Error will be thrown when this thread is killed by keyboard interruption to stop the recording
        except Exception as e:
            pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AI Meeting Transcription with Rolling Summaries')
    parser.add_argument(
        '-bs',
        '--buffer_size',
        help=(
            'How many seconds of audio per buffer. ' 
            'The smaller this value, the more real time the transcription, but at the cost of accuracy. '
            'The larger this value, the more accurate the transcription, but it is slower. '
            'Defaults to 10 seconds.'
        ),
        nargs='?',
        default=10,
        type=int,
    )
    parser.add_argument(
        '-sb',
        '--summary_buffers',
        help=(
            'How many buffers to use per summary. '
            'The smaller this value, the narrower the resulting summary. '
            'The larger this value, the broader the resulting summary. '
            'Defaults to 6, which produces a summary every 60 seconds if buffer_size = 10.'
        ),
        nargs='?',
        default=6,
        type=int,
    )
    parser.add_argument(
        '-wt',
        '--word_to_track',
        help='If a word is provided, it will be highlighted when printed to the terminal.',
        nargs='?',
        type=str,
    )
    args = vars(parser.parse_args())

    # It is best to leave sample rate (samples_per_second) alone, as Whisper was trained on a sample rate of 16,000
    params = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "nbr_bits_per_sample": pyaudio.paInt16,
        "channels": 1,
        "samples_per_second": 16_000,
        "record_in_chunks_of_this_many_samples": 2_048,
    }

    # Initialize models for ASR and abstractive summarization, the PortAudio interface, and colorama
    model = WhisperModel("small.en", device=params['device'], compute_type="float32")
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    mic = pyaudio.PyAudio()
    init(autoreset=True)  # colorama

    # Define a keyboard listener and an audio stream
    listener = Listener(on_press=on_press, on_release=on_release)
    stream = mic.open(
        format=params['nbr_bits_per_sample'],
        channels=params['channels'],
        rate=params['samples_per_second'],
        input=True,
        output=True,
        frames_per_buffer=params['record_in_chunks_of_this_many_samples']
    )

    # Start the stream and listener threads
    stream.start_stream()
    listener.start()

    print(
        "\n==============================\nRecording started. "
        "Press ESC to stop.\n==============================\n"
    )

    # Start recording in a new thread
    raw_recording, recorded_arrays, transcribed_text = [], [], []
    Thread(
        target=record,
        args=(
            params['samples_per_second'],
            params['record_in_chunks_of_this_many_samples'],
            args['buffer_size'],
            args['summary_buffers'],
            raw_recording,
            recorded_arrays,
            transcribed_text,
            args['word_to_track'],
        ),
        name='record_loop',
        daemon=True,
    ).start()

    # Kill the record_loop thread with the keyboard listener thread
    listener.join()

    # Close the recording stream and terminate PortAudio
    stream.stop_stream()
    stream.close()
    mic.terminate()

    print(
        "\n==============================\nRecording stopped. "
        "\nIgnore any errors after the last transcription that you see, as they were caused by "
        "forcefully terminating the threads while new data was being recorded and processed."
        "\n==============================\n"
    )

    # save the recording
    print("Saving recording to .wav")
    save_recording_to_wav(
        recording=raw_recording,
        sample_rate=params['samples_per_second'],
        file_path="speaker_analysis/data/meeting_recording.wav"
    )

    # save the transcription
    if len(transcribed_text) > 0:
        print("Saving transcription to .txt")
        with open("transcript_rag/data/meeting_transcript.txt", "w") as f:
            f.write(" ".join(transcribed_text))
    else:
        print(
            "There was no transcription.  If there should have been one, please troubleshoot by:\n"
            "1. Ensuring that the model had enough time to transcribe what was recorded before the Esc key was pressed."
            "2. Ensuring that the mic was on and speakers were speaking loud enough to be captured in the .wav file."
        )
