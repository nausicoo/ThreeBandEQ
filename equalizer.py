"""
DSP learning log:

Day 2:

Raw audio loading and signal diagnostics

After generating a controlled synthetic WAV file on day 1, I spent day 2
building and dissecting the foundational equalizer class. Instead of rushing
straight into filters, I took time to understand how NumPy stores audio arrays
in memory and how SoundFile normalizes sample values.

Goal:
Implement the base `ThreeBandEqualizer` class to load WAV files, it detects
mono vs. stereo channel structures, and inspects acoustic properties
(sample rate, nyquist limit, duration and peak amplitude).

Key takeaways:
- Unlike symbolic formats like MIDI, raw digital audio is simply a 1D (mono)
  or 2D (stereo) array of floats normalized between -1.0 and 1.0.
- At 44,100 hz, just 3 seconds of mono audio already requires 132,300 samples.
- The Nyquist limit (sample_rate / 2) defines the highest frequency the file
  can represent and sets the boundary for upcoming crossover filters.
- Casting np.max() explicitly to Python's native float() prevents static typing
  conflicts when formatting output values.

"""

import numpy as np #Numpy works with optimized arrays in C that allows to perform mathematical operations on the entire audio signal
import soundfile as sf #Bridge between Python and the hard drive. It can read real audio files (.wav, .flac, etc.) and convert them into NumPy arrays,
# and it can do the other way around too, saving the arrays as a .wav file


class ThreeBandEqualizer:
    def __init__(self, file_path: str): # Initialize the equalizer with a file path
        self.file_path = file_path # Store the file path in an instance variable for later use
        self.audio_data = None # Initialize audio_data to None; it will hold the loaded audio signal as a NumPy array,
        # it doesn't read the file yet, just prepares the variable to hold the data once loaded,
        # the same with sample_rate and channels, which will be set when the audio file is loaded
        self.sample_rate = 0
        self.channels = 0

    def load_audio(self) -> bool:
        """Loads a WAV file into a Numpy array and stores signal data."""
        try:
            # Unlike MIDI, which stores symbolic instructions like "play C4",
            # raw audio is just a massive array of numbers representing air pressure
            # variations over time. soundfile.read() automatically normalizes these
            # values into floating-point numbers between -1.0 and 1.0.
            self.audio_data, self.sample_rate = sf.read(self.file_path) # Read the audio file and store the data and sample rate, it converts the audio file into a NumPy array,
            # where each element represents a sample of the audio signal, and the sample rate is stored for later use in processing or playback.

            # Checking array dimensions matters before doing any DSP math.
            # A 1D array (N,) means Mono. A 2D array (N, 2) means Stereo (L/R).
            # I need to track this so my future filters don't crash on stereo files.
            if self.audio_data.ndim == 1: # If the audio data is a 1D array, it means the audio is mono (single channel), so we set channels to 1.
                self.channels = 1
            else:
                self.channels = self.audio_data.shape[1] # If the audio data is a 2D array, it means the audio is stereo (two channels), so we set channels to the second dimension of the array

            return True

        except FileNotFoundError:
            print(f"Error: File '{self.file_path}' not found.")
            return False
        except RuntimeError as e: # Catching a RuntimeError, which can occur if the file format is unsupported or if there's an issue reading the file, and prints the error message
            print(f"Error reading audio file: {e}") # The error message provides details about what went wrong during the file reading process, which can help in diagnosing the issue.
            return False

    def inspect_signal(self):
        """Prints core mathematical properties of the loaded audio signal."""
        if self.audio_data is None: # If audio_data is None, it means no audio has been loaded yet, so we print a message indicating that the user should call load_audio() first to load an audio file before inspecting its properties.
            print("No audio loaded. Call load_audio() first.")
            return # Return early if no audio is loaded

        total_samples = len(self.audio_data) # Calculates the number of samples in the audio signal, which is simply the length of the audio_data array. This gives an idea of how many discrete points in time there are
        duration_sec = total_samples / self.sample_rate # Calculates the duration of the audio signal in seconds by dividing the total number of samples by the sample rate. This gives the total time span of the audio signal, which is useful for understanding how long the audio will play.
        peak_amplitude = float(np.max(np.abs(self.audio_data))) # Calculates the peak amplitude of the audio signal by taking the maximum absolute value of the audio_data array. This gives an idea of the loudest point in the audio signal, which is important for understanding the dynamic range and ensuring that the signal does not clip

        # The "Nyquist frequency" is half the sample rate (e.g., 22,050 Hz for 44.1 kHz).
        # It represents the highest frequency this digital file can accurately capture.
        # I'll need this limit when setting up crossover frequencies for the filters.
        nyquist_freq = self.sample_rate / 2

        print("\n--- Audio signal diagnostic ---")
        print(f"File: {self.file_path}") # Print the file path of the loaded audio signal, which helps identify which audio file is being analyzed, especially when working with multiple files.
        print(f"Sample rate: {self.sample_rate} Hz (Nyquist limit: {nyquist_freq} Hz)") # Print the sample rate of the audio signal in Hertz, along with the Nyquist limit
        print(f"Channels: {self.channels} ({'Mono' if self.channels == 1 else 'Stereo'})") # Print the number of channels in the audio signal, indicating whether it is mono or stereo
        print(f"Array shape: {self.audio_data.shape}") # Print the shape of the audio_data array, which indicates the dimensions of the data
        print(f"Total samples (per channel): {total_samples}") # Print the total number of samples in the audio signal, which is useful for understanding the length of the signal
        print(f"Duration: {duration_sec:.2f} seconds") # Print the duration of the audio signal in seconds, which is useful for understanding how long the audio will play
        print(f"Peak amplitude: {peak_amplitude:.4f} (safe range: 0.0 to 1.0)") # Print the peak amplitude of the audio signal, along with a note about the safe range for audio signals, which is typically between 0.0 and 1.0 to avoid clipping and distortion
        print(f"First 5 raw samples: {self.audio_data[:5]}") # Print the first five samples of the audio_data array, which provides a quick look at the raw numerical values representing the audio signal. This can be useful for debugging or verifying that the audio data has been loaded correctly.

if __name__ == "__main__":
    eq = ThreeBandEqualizer("test_signal.wav")
    if eq.load_audio():
        eq.inspect_signal()