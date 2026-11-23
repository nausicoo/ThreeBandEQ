"""
DSP learning log:

Day 2: Raw audio loading and signal diagnostics

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

Day 3: Butterworth Crossover Filters & Band Splitting

Implemented the `split_bands()` method using `scipy.signal` to separate the
synthetic 3-frequency test signal into isolated low, mid, and high bands
with crossover frequencies at 250 Hz and 3000 Hz.

Key takeaways:
- Crossover frequencies must be normalized against the Nyquist limit
  (cutoff / nyquist) to match SciPy's [0, 1] digital frequency range.
- Used second-order sections (output='sos') instead of classic (b, a)
  coefficients to maintain numerical stability at lower cutoff frequencies.
- Applied zero-phase filtering with `sosfiltfilt(..., axis=0)` to prevent
  phase distortion and keep the processing compatible with mono and stereo arrays.
- Band separation succeeded: the combined peak of 0.8634 split into
  0.3126 (low), 0.3438 (mid), and 0.3334 (high), closely matching the
  original 0.3000 amplitude of each sine wave.
- The slight overshoot above 0.3000 is caused by IIR filter boundary transients
  (edge ringing) during forward-backward filtering.

Day 4: dB Gain scaling, band recombination & clipping protection

Added decibel-to-linear amplitude conversion (`db_to_linear()`), vectorized band
mixing with automatic clipping protection (`apply_eq()`), and WAV export
(`export_audio()`).

Key takeaways:
- Human loudness perception is logarithmic, so EQ gains in dB are converted to
  linear amplitude multipliers via g = 10^(dB / 20) (e.g., +6 dB ≈ 1.9953x,
  +12 dB ≈ 3.9811x, -24 dB ≈ 0.0631x).
- Flat-response test (0 dB across all bands) reconstructed the signal with a
  0.8870 peak vs. the original 0.8634, confirming phase-coherent recombination
  with minor IIR edge-transient buildup.
- Moderate EQ test (+6 dB Low, -24 dB Mid, 0 dB High) produced a safe 0.9285
  peak without triggering clipping protection.
- Stress test (+12 dB Low, -24 dB Mid, 0 dB High) pushed the raw summed peak
  to 1.5493, successfully triggering automatic proportional peak scaling
  (factor: 0.6390) to bring the exported peak down to 0.9900 without hard
  digital clipping or altering relative band balance.

"""

import numpy as np #Numpy works with optimized arrays in C that allows to perform mathematical operations on the entire audio signal
import soundfile as sf #Bridge between Python and the hard drive. It can read real audio files (.wav, .flac, etc.) and convert them into NumPy arrays,
# and it can do the other way around too, saving the arrays as a .wav file
from scipy.signal import butter, sosfiltfilt # The `butter` function designs Butterworth filters, and `sosfiltfilt` applies these filters to the audio signal using second-order sections for numerical stability.


class ThreeBandEqualizer:
    def __init__(self, file_path: str): # Initialize the equalizer with a file path
        self.file_path = file_path # Store the file path in an instance variable for later use
        self.audio_data = None # Initialize audio_data to None; it will hold the loaded audio signal as a NumPy array,
        # it doesn't read the file yet, just prepares the variable to hold the data once loaded,
        # the same with sample_rate and channels, which will be set when the audio file is loaded
        self.sample_rate = 0
        self.channels = 0
        self.low_band = None # Initialize low_band to None; it will hold the low-frequency band after filtering
        self.mid_band = None # The same applies to mid_band and high_band, which will hold the mid and high-frequency bands after filtering
        self.high_band = None
        self.processed_audio = None # Initialize processed_audio to None; it will hold the final equalized audio signal after applying EQ settings

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

        # The "Nyquist frequency" is half the sample rate (e.g. 22,050 Hz for 44.1 khz)
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

    def split_bands(self, low_cutoff: float = 250.0, high_cutoff: float = 3000.0, order: int = 4):
        """Splits the loaded audio into low, mid, and high frequency bands using butterworth filters."""
        if self.audio_data is None: # If audio_data is None, it means no audio has been loaded yet, so we print a message indicating that the user should call load_audio() first to load an audio file before attempting to split the bands.
            print("No audio loaded. Call load_audio() first.")
            return

        nyquist = self.sample_rate / 2 # Calculate the Nyquist frequency, which is half the sample rate. This value is used to normalize the cutoff frequencies for the filter design, ensuring that they are within the valid range for digital filtering.

        # Normalize crossover frequencies to the [0, 1] nyquist range
        low_norm = low_cutoff / nyquist # Normalize the low cutoff frequency by dividing it by the Nyquist frequency. This converts the cutoff frequency into a normalized value between 0 and 1
        high_norm = high_cutoff / nyquist # Same normalization is done for the high cutoff frequency, ensuring that both cutoff frequencies are appropriately scaled for the filter design process

        # Design the three butterworth filters using Second-order sections (SOS)
        sos_low = butter(order, low_norm, btype='lowpass', output='sos') # Design a low-pass Butterworth filter using the specified order and normalized low cutoff frequency. The output is in second-order sections (SOS) format, which is numerically stable for filtering operations.
        sos_mid = butter(order, [low_norm, high_norm], btype='bandpass', output='sos') # Same process is done for the mid-band filter and the high-band filter, ensuring that all three filters are designed to isolate their respective frequency bands effectively.
        sos_high = butter(order, high_norm, btype='highpass', output='sos')

        # Apply zero-phase filtering along axis=0 (time axis) so it works for mono and stereo
        self.low_band = sosfiltfilt(sos_low, self.audio_data, axis=0) # Apply the low-pass filter to the audio data using zero-phase filtering, which prevents phase distortion in the filtered signal. The filtering is applied along the time axis (axis=0), ensuring that it works correctly for both mono and stereo audio signals.
        self.mid_band = sosfiltfilt(sos_mid, self.audio_data, axis=0) # # Same process is done for the mid-band and the high-band filters, ensuring that all three frequency bands are separated effectively without introducing phase distortion.
        self.high_band = sosfiltfilt(sos_high, self.audio_data, axis=0)

        print(f"\n--- Band separation complete ({low_cutoff} Hz / {high_cutoff} Hz, order {order}) ---")
        print(f"Low band peak:  {float(np.max(np.abs(self.low_band))):.4f}")
        print(f"Mid band peak:  {float(np.max(np.abs(self.mid_band))):.4f}")
        print(f"High band peak: {float(np.max(np.abs(self.high_band))):.4f}")

    @staticmethod
    def db_to_linear(db: float) -> float:
        """Converts decibels (dB) to a linear amplitude multiplier."""
        # This method converts a gain value in decibels (dB) to a linear amplitude multiplier.
        # The formula used is 10^(db/20), which is derived from the logarithmic relationship between decibels and linear amplitude.
        # This conversion is necessary because audio processing often requires linear gain values for mathematical operations,
        # while user interfaces and specifications commonly use decibel values for their logarithmic representation of sound intensity.
        return 10.0 ** (db / 20.0)

    def apply_eq( # This method applies the equalization settings to the separated frequency bands.
            # It takes gain values in decibels for the low, mid, and high bands, converts them to linear multipliers, and then combines the bands back into a single audio signal.
            # It also includes an option to prevent clipping by scaling down the signal if the peak amplitude exceeds 1.0.
            self,
            low_db: float = 0.0, # The default gain for the low band is set to 0.0 dB, meaning no change in amplitude.
            # The user can specify a different gain value in decibels to boost or cut the high, mid, and low frequencies.
            mid_db: float = 0.0,
            high_db: float = 0.0,
            prevent_clipping: bool = True, # The prevent_clipping parameter is a boolean flag that determines whether the method should automatically scale down the combined audio signal to prevent clipping.
            # If set to True, the method will check the peak amplitude of the mixed signal and scale it down if it exceeds 1.0, ensuring that the output remains within the safe range for audio signals.
    ):
        """Applies dB gains to each band, recombines them, and protects against clipping."""
        if self.low_band is None or self.mid_band is None or self.high_band is None:
            print("Bands not split yet. Call split_bands() first.") # If any of the frequency bands (low, mid, or high) have not been split yet (they are still None),
            # it means that the user has not called the split_bands() method to separate the audio into its respective frequency bands.
            # In this case, the method prints a message indicating that the user should call split_bands() first before applying equalization,
            # and then it returns early without making any changes to the audio signal.
            return

        low_gain = self.db_to_linear(low_db) # This line converts the gain value for the low band from decibels to a linear amplitude multiplier using the db_to_linear() method.
        mid_gain = self.db_to_linear(mid_db)
        high_gain = self.db_to_linear(high_db)

        mixed = (
                (self.low_band * low_gain)
                + (self.mid_band * mid_gain)
                + (self.high_band * high_gain)
        ) # This line combines the three frequency bands (low, mid, and high) into a single audio signal by applying the respective linear gain multipliers to each band and summing them together.

        raw_peak = float(np.max(np.abs(mixed))) # This line calculates the peak amplitude of the mixed audio signal by taking the maximum absolute value of the combined signal.
        # It converts the result to a float for consistent formatting and further processing.

        if prevent_clipping and raw_peak > 1.0: # If the prevent_clipping flag is set to True and the raw peak amplitude of the mixed signal exceeds 1.0,
            # it indicates that the audio signal is at risk of clipping (distortion due to exceeding the maximum amplitude range).
            scale_factor = 0.99 / raw_peak
            mixed = mixed * scale_factor
            print(
                f"\n[Clipping protection] Raw peak ({raw_peak:.4f}) exceeded 1.0. "
                f"Scaled down to 0.9900 (factor: {scale_factor:.4f})."
            )

        self.processed_audio = mixed
        final_peak = float(np.max(np.abs(self.processed_audio))) # This line calculates the peak amplitude of the processed audio signal after applying equalization and any necessary clipping protection.
        # It takes the maximum absolute value of the processed audio array and converts it to a float for consistent formatting and further processing.

        print(f"\n--- EQ Applied (Low: {low_db:+.1f} dB, Mid: {mid_db:+.1f} dB, High: {high_db:+.1f} dB) ---")
        print(f"Linear gains -> Low: {low_gain:.4f}x | Mid: {mid_gain:.4f}x | High: {high_gain:.4f}x")
        print(f"Final mixed peak: {final_peak:.4f}")

    def export_audio(self, output_path: str):
        """Exports the equalized audio array to a WAV file."""
        if self.processed_audio is None:
            print("No processed audio to export. Call apply_eq() first.")
            return

        sf.write(output_path, self.processed_audio, self.sample_rate) # This line uses the soundfile library to write the processed audio data to a WAV file at the specified output path.
        print(f"Exported equalized audio to '{output_path}'")

if __name__ == "__main__":
    eq = ThreeBandEqualizer("test_signal.wav")
    if eq.load_audio():
        eq.inspect_signal()
        eq.split_bands(low_cutoff=250.0, high_cutoff=3000.0, order=4)

        # Test 1: Flat response check (all bands at 0 dB should reconstruct the original signal)
        eq.apply_eq(low_db=0.0, mid_db=0.0, high_db=0.0)

        # Test 2: Actual EQ curve (boost lows, heavily cut mids, keep highs neutral)
        eq.apply_eq(low_db=12.0, mid_db=-24.0, high_db=0.0)
        eq.export_audio("output_eq_test.wav")