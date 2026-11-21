"""
DSP learning log:

Day 1:

Synthetic test signal generator:

This is my first experiment with digital audio synthesis in python.
I am documenting what each function does as I learn the basics of DSP from scratch.

Goal of this script:
Generate a controlled WAV file combining three sine waves (100 Hz, 1000 Hz,
and 6000 Hz). Isolating frequencies will make it easier to verify
whether my crossover filters will actually isolate the low, mid, and high bands

"""

import numpy as np #Numpy works with optmized arrays in C that allows to perform mathematical operations on the entire audio signal all at once
import soundfile as sf #Bridge between Python and the hard drive. It can read real audio files (.wav, .flac, etc.) and convert them into NumPy arrays,
# and it can do the other way around too, saving the arrays as a .wav file


sample_rate = 44100 #Defines the sampling frequency.
duration = 3.0  # seconds

# It creates a t array that represents time from 0 to the duration of the signal, with a number of samples equal to sample_rate * duration.
# np.linspace() generates evenly spaced values over a specified interval.
# 0, duration defines the start and end of the interval (0 to 3 seconds).
# int(sample_rate * duration) calculates the total number of samples needed for the specified duration at the given sample rate.
# It has to be an integer because you can't have a fraction of a sample.
# endpoint=False ensures that the endpoint (3 seconds) is not included in the array, which is important for generating a continuous signal without overlap at the end.
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# Combine three sine waves: Low (100 Hz), Mid (1000 Hz), High (6000 Hz)
# The amplitude of each sine wave is set to 0.3 to avoid clipping when they are combined.
# np.sin() generates the sine wave values for each frequency over the time array t.
# The formula 2 * np.pi * frequency * t converts the frequency into radians per second, which is necessary for generating the sine wave.
# 100, 1000, and 6000 are the frequencies of the low, mid, and high sine waves, respectively.
# t evaluates the sine function at each point in time, creating a continuous wave for each frequency.
low_wave = 0.3 * np.sin(2 * np.pi * 100 * t)
mid_wave = 0.3 * np.sin(2 * np.pi * 1000 * t)
high_wave = 0.3 * np.sin(2 * np.pi * 6000 * t)

#In digital audio, combining multiple signals is done by adding their corresponding sample values together.
# This is a straightforward operation in NumPy, which allows for element-wise addition of arrays.
# The resulting array, combined_signal, contains the sum of the low, mid, and high frequency sine waves at each point in time.
# In phyics , this is analogous to the principle of wave superposition, where the resultant wave at any point is the sum of the individual waves at that point.
combined_signal = low_wave + mid_wave + high_wave

# Save the combined signal to a .wav file
sf.write("test_signal.wav", combined_signal, sample_rate)
print("Generated test_signal.wav (100 Hz + 1000 Hz + 6000 Hz)")