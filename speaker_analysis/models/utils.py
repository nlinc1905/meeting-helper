import numpy as np
import typing as t
import plotly
from itertools import cycle


def audio_to_numpy(audio_buffer: bytes) -> np.ndarray:
    """
    Converts audio buffer to a float32 numpy array.
    The conversion normalizes the float32 array so that values are between -1.0 and 1.0 by
    dividing by 2**15.

    The audio is assumed to have 1 channel so that it does not need to be flattened.
    If the audio has 2 channels, the resulting array will still be 1-dimensional but the
    array values will be the interleaved channel values.
    """
    max_int16 = 2 * 15
    buffer = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / max_int16
    return buffer


def assign_color_to_category(categories: t.Iterable, hardcoded_categories: dict = None) -> dict:
    """
    Assigns colors to categories, using Plotly's default color scheme.
    Loops back to the beginning if the number of categories > number of colors
    in the default scheme.
    """
    if not hardcoded_categories:
        hardcoded_categories = {}
    color_options = cycle(plotly.colors.DEFAULT_PLOTLY_COLORS)
    color_assignments = {c: next(color_options) for c in categories if c not in list(hardcoded_categories)}
    color_assignments.update(hardcoded_categories)
    return color_assignments
