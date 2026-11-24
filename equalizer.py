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

Day 5: Spectral analysis & filter response visualization

Added `plot_filter_responses()` (using `scipy.signal.sosfreqz`) and
`plot_spectrum_comparison()` (using a Hann-windowed `rfft`) to visually validate
the crossover behavior and EQ spectral modifications via `matplotlib`.

Key takeaways:
- Applying 4th-order Butterworth filters with zero-phase `sosfiltfilt` squares
  the magnitude response (|H(f)|^2), yielding effective 8th-order
  Linkwitz-Riley-like crossovers meeting at -6 dB with a nearly flat combined sum.
- Applying a Hann window before `np.fft.rfft()` minimizes spectral leakage and
  clearly resolves the 100 Hz, 1000 Hz, and 6000 Hz tones in dB FS.
- The FFT plot visually confirmed both the +12 dB bass boost / -24 dB mid cut
  and the uniform -3.89 dB shift across all bands caused by the proportional
  clipping protection factor (0.6390), proving relative band balance is preserved.

"""

import numpy as np  # NumPy works with optimized arrays in C that allow performing mathematical operations on the entire audio signal
import soundfile as sf  # Bridge between Python and the hard drive; reads real audio files (.wav, .flac, etc.) into NumPy arrays and saves arrays back as .wav files
import matplotlib.pyplot as plt  # Plotting library used to draw and save the filter frequency response and FFT spectrum comparison graphs
from scipy.signal import butter, sosfiltfilt, sosfreqz  # `butter` designs butterworth filters, `sosfiltfilt` applies them with zero phase distortion, and `sosfreqz` calculates their frequency response


class ThreeBandEqualizer:
    def __init__(self, file_path: str):  # Initialize the equalizer with a file path
        self.file_path = file_path  # Store the file path in an instance variable for later use
        self.audio_data = None  # Initialize audio_data to None; it will hold the loaded audio signal as a NumPy array once loaded
        # It does not read the file yet, just prepares the variables to hold the data once load_audio() runs
        self.sample_rate = 0  # Will store the sampling frequency in Hz (e.g., 44100)
        self.channels = 0  # Will store the number of audio channels (1 for mono, 2 for stereo)
        self.low_band = None  # Will hold the isolated low-frequency band array after filtering
        self.mid_band = None  # Will hold the isolated mid-frequency band array after filtering
        self.high_band = None  # Will hold the isolated high-frequency band array after filtering
        self.processed_audio = None  # Will hold the final equalized and mixed audio array after applying EQ settings
        self.sos_low = None  # Stores the low-pass filter coefficients (second-order sections) so we can plot the filter curve later
        self.sos_mid = None  # Stores the band-pass filter coefficients for plotting
        self.sos_high = None  # Stores the high-pass filter coefficients for plotting
        self.low_cutoff = 0.0  # Stores the low crossover frequency in Hz (initialized as float 0.0 to avoid static typing warnings with None)
        self.high_cutoff = 0.0  # Stores the high crossover frequency in Hz
        self.last_gains_db = None  # Stores a tuple (low_db, mid_db, high_db) of the last applied EQ settings to display in the FFT plot title

    def load_audio(self) -> bool:
        """Loads a WAV file into a Numpy array and stores signal data."""
        try:
            # Unlike MIDI, which stores symbolic instructions like "play C4",
            # raw audio is just a massive array of numbers representing air pressure
            # variations over time. soundfile.read() automatically normalizes these
            # values into floating-point numbers between -1.0 and 1.0.
            self.audio_data, self.sample_rate = sf.read(self.file_path)  # Reads the audio file into a NumPy array and extracts its sample rate

            # Checking array dimensions matters before doing any DSP math.
            # A 1D array (N,) means mono. A 2D array (N, 2) means stereo (L/R).
            # We track this so the filters do not crash on stereo files.
            if self.audio_data.ndim == 1:  # If the audio data is a 1D array, it is mono (single channel)
                self.channels = 1
            else:
                self.channels = self.audio_data.shape[1]  # If it is a 2D array, the second dimension gives the number of channels (e.g., 2 for stereo)

            return True

        except FileNotFoundError:  # Runs if the specified file path does not exist on disk
            print(f"Error: File '{self.file_path}' not found.")
            return False
        except RuntimeError as e:  # Runs if the file format is unsupported or corrupted when reading
            print(f"Error reading audio file: {e}")
            return False

    def inspect_signal(self):
        """Prints core mathematical properties of the loaded audio signal."""
        if self.audio_data is None:  # Guard clause: exits early if load_audio() has not been called yet
            print("No audio loaded. Call load_audio() first.")
            return

        total_samples = len(self.audio_data)  # Total number of discrete time points (samples per channel)
        duration_sec = total_samples / self.sample_rate  # Total audio length in seconds (samples divided by samples per second)
        peak_amplitude = float(np.max(np.abs(self.audio_data)))  # Loudest absolute sample value, cast to native Python float to avoid typing issues

        # The nyquist frequency is half the sample rate (e.g., 22,050 Hz for 44.1 kHz).
        # It represents the highest frequency this digital file can accurately capture,
        # and it sets the reference limit for normalizing crossover frequencies.
        nyquist_freq = self.sample_rate / 2

        print("\n--- Audio signal diagnostic ---")
        print(f"File: {self.file_path}")  # Prints the loaded file name/path
        print(f"Sample rate: {self.sample_rate} Hz (Nyquist limit: {nyquist_freq} Hz)")  # Prints sample rate and nyquist frequency
        print(f"Channels: {self.channels} ({'Mono' if self.channels == 1 else 'Stereo'})")  # Prints channel count and mono/stereo label
        print(f"Array shape: {self.audio_data.shape}")  # Prints the exact dimensions of the NumPy array
        print(f"Total samples (per channel): {total_samples}")  # Prints how many samples exist per channel
        print(f"Duration: {duration_sec:.2f} seconds")  # Prints total duration formatted to 2 decimal places
        print(f"Peak amplitude: {peak_amplitude:.4f} (safe range: 0.0 to 1.0)")  # Prints peak level formatted to 4 decimal places
        print(f"First 5 raw samples: {self.audio_data[:5]}")  # Prints the first 5 raw sample numbers to verify the data loaded properly

    def split_bands(self, low_cutoff: float = 250.0, high_cutoff: float = 3000.0, order: int = 4):
        """Splits the loaded audio into low, mid, and high frequency bands using butterworth filters."""
        if self.audio_data is None:  # Guard clause: prevents filtering if no audio is loaded yet
            print("No audio loaded. Call load_audio() first.")
            return

        nyquist = self.sample_rate / 2  # Calculates the nyquist frequency (half the sample rate) used to normalize cutoffs

        # Normalize crossover frequencies to SciPy's [0, 1] nyquist range
        low_norm = low_cutoff / nyquist  # Converts the low cutoff Hz into a normalized value between 0 and 1
        high_norm = high_cutoff / nyquist  # Converts the high cutoff Hz into a normalized value between 0 and 1

        self.low_cutoff = low_cutoff  # Saves the low crossover Hz on the instance so plot methods can draw reference lines
        self.high_cutoff = high_cutoff  # Saves the high crossover Hz on the instance

        # Design the three butterworth filters using second-order sections (sos) for numerical stability
        self.sos_low = butter(order, low_norm, btype='lowpass', output='sos')  # Designs the low-pass filter (keeps frequencies below low_cutoff)
        self.sos_mid = butter(order, [low_norm, high_norm], btype='bandpass', output='sos')  # Designs the band-pass filter (keeps frequencies between low_cutoff and high_cutoff)
        self.sos_high = butter(order, high_norm, btype='highpass', output='sos')  # Designs the high-pass filter (keeps frequencies above high_cutoff)

        # Apply zero-phase filtering along axis=0 (time axis) so it works for both mono and stereo without phase distortion
        self.low_band = sosfiltfilt(self.sos_low, self.audio_data, axis=0)  # Runs the low-pass filter forward and backward over the audio
        self.mid_band = sosfiltfilt(self.sos_mid, self.audio_data, axis=0)  # Runs the band-pass filter forward and backward over the audio
        self.high_band = sosfiltfilt(self.sos_high, self.audio_data, axis=0)  # Runs the high-pass filter forward and backward over the audio

        print(f"\n--- Band separation complete ({low_cutoff} Hz / {high_cutoff} Hz, order {order}) ---")
        print(f"Low band peak:  {float(np.max(np.abs(self.low_band))):.4f}")  # Prints the highest peak inside the isolated low band
        print(f"Mid band peak:  {float(np.max(np.abs(self.mid_band))):.4f}")  # Prints the highest peak inside the isolated mid band
        print(f"High band peak: {float(np.max(np.abs(self.high_band))):.4f}")  # Prints the highest peak inside the isolated high band

    @staticmethod
    def db_to_linear(db: float) -> float:
        """Converts decibels (dB) to a linear amplitude multiplier."""
        # This method does not need `self` because it is a pure math conversion.
        # Uses 10^(db/20) because human hearing and audio interfaces work in logarithmic dB,
        # while NumPy array math needs a linear amplitude multiplier (e.g., 0 dB -> 1.0x, +6 dB -> ~2.0x).
        return 10.0 ** (db / 20.0)

    def apply_eq(
            self,
            low_db: float = 0.0,  # Gain in dB for the low band (0.0 dB means no change)
            mid_db: float = 0.0,  # Gain in dB for the mid band
            high_db: float = 0.0,  # Gain in dB for the high band
            prevent_clipping: bool = True,  # Safety toggle: if True, scales down the final mix when peak exceeds 1.0
    ):
        """Applies dB gains to each band, recombines them, and protects against clipping."""
        if self.low_band is None or self.mid_band is None or self.high_band is None:  # Guard clause: ensures split_bands() was called before mixing
            print("Bands not split yet. Call split_bands() first.")
            return

        low_gain = self.db_to_linear(low_db)  # Converts low band dB gain to a linear multiplier
        mid_gain = self.db_to_linear(mid_db)  # Converts mid band dB gain to a linear multiplier
        high_gain = self.db_to_linear(high_db)  # Converts high band dB gain to a linear multiplier

        # Vectorized NumPy math: multiplies every sample in each band by its gain and sums all three bands back together
        mixed = (
                (self.low_band * low_gain)
                + (self.mid_band * mid_gain)
                + (self.high_band * high_gain)
        )

        raw_peak = float(np.max(np.abs(mixed)))  # Finds the maximum absolute sample amplitude in the recombined signal

        if prevent_clipping and raw_peak > 1.0:  # Triggers only if clipping protection is enabled and the mix exceeds the digital ceiling of 1.0
            scale_factor = 0.99 / raw_peak  # Calculates the exact ratio needed to pull the highest peak down to 0.99
            mixed = mixed * scale_factor  # Attenuates the entire signal uniformly so relative balance between bands stays intact
            print(
                f"\n[Clipping protection] Raw peak ({raw_peak:.4f}) exceeded 1.0. "
                f"Scaled down to 0.9900 (factor: {scale_factor:.4f})."
            )

        self.processed_audio = mixed  # Stores the final equalized audio array in the object for export or plotting
        self.last_gains_db = (low_db, mid_db, high_db)  # Saves the applied dB gains so plot_spectrum_comparison() can show them in its title
        final_peak = float(np.max(np.abs(self.processed_audio)))  # Measures the final peak after optional clipping protection

        print(f"\n--- EQ Applied (Low: {low_db:+.1f} dB, Mid: {mid_db:+.1f} dB, High: {high_db:+.1f} dB) ---")  # `+` forces showing the sign (+ or -)
        print(f"Linear gains -> Low: {low_gain:.4f}x | Mid: {mid_gain:.4f}x | High: {high_gain:.4f}x")  # Shows the linear multipliers used
        print(f"Final mixed peak: {final_peak:.4f}")  # Confirms the final output level is inside the safe [0.0, 1.0] range

    def export_audio(self, output_path: str):
        """Exports the equalized audio array to a WAV file."""
        if self.processed_audio is None:  # Guard clause: ensures apply_eq() has run before trying to write to disk
            print("No processed audio to export. Call apply_eq() first.")
            return

        sf.write(output_path, self.processed_audio, self.sample_rate)  # Writes the processed NumPy array to a WAV file using the original sample rate
        print(f"Exported equalized audio to '{output_path}'")

    def plot_filter_responses(self, save_path: str | None = "filter_response.png"):
        """Plots the effective zero-phase magnitude response of the 3 butterworth bands."""
        if self.sos_low is None or self.sos_mid is None or self.sos_high is None:  # Guard clause: requires filters to be designed first
            print("Filters not designed yet. Call split_bands() first.")
            return

        # Evaluates the theoretical frequency response of each filter across 8192 frequency points (high resolution keeps low frequencies smooth on a log axis)
        # Passing fs=self.sample_rate returns `freqs` in Hz instead of radians; `h_*` are complex arrays holding magnitude and phase
        freqs, h_low = sosfreqz(self.sos_low, worN=8192, fs=self.sample_rate)
        _, h_mid = sosfreqz(self.sos_mid, worN=8192, fs=self.sample_rate)  # `_` discards the duplicate freqs array
        _, h_high = sosfreqz(self.sos_high, worN=8192, fs=self.sample_rate)

        # Because sosfiltfilt filters the signal twice (forward + backward), the effective magnitude gets squared: |H(f)|^2
        # np.abs() extracts the magnitude from each complex number in h_*
        mag_low = np.abs(h_low) ** 2
        mag_mid = np.abs(h_mid) ** 2
        mag_high = np.abs(h_high) ** 2

        # Converts linear magnitudes to decibels via 20 * log10(mag); np.maximum(..., 1e-10) sets a floor of -200 dB to prevent log10(0) errors
        db_low = 20 * np.log10(np.maximum(mag_low, 1e-10))
        db_mid = 20 * np.log10(np.maximum(mag_mid, 1e-10))
        db_high = 20 * np.log10(np.maximum(mag_high, 1e-10))

        # Combined flat-EQ response: because zero-phase filtering removes phase shift, the effective magnitudes add directly
        db_sum = 20 * np.log10(np.maximum(mag_low + mag_mid + mag_high, 1e-10))

        plt.figure(figsize=(10, 5))  # Creates a new plot canvas of 10x5 inches
        plt.semilogx(freqs, db_low, label="Low band (Low-pass)", linewidth=2)  # Plots low-pass curve on a logarithmic X axis (frequency) and linear Y axis (dB)
        plt.semilogx(freqs, db_mid, label="Mid band (Band-pass)", linewidth=2)  # Plots band-pass curve
        plt.semilogx(freqs, db_high, label="High band (High-pass)", linewidth=2)  # Plots high-pass curve
        plt.semilogx(freqs, db_sum, label="Combined flat sum", color="black", linestyle="--", alpha=0.7)  # Plots the dashed black sum line to verify flatness

        # Draws vertical dotted reference lines at the low and high crossover frequencies
        plt.axvline(self.low_cutoff, color="gray", linestyle=":",
                    label=f"Crossovers ({self.low_cutoff:.0f} / {self.high_cutoff:.0f} Hz)")
        plt.axvline(self.high_cutoff, color="gray", linestyle=":")
        plt.axhline(-6.0, color="red", linestyle=":", alpha=0.5, label="-6 dB Crossover point")  # Draws a horizontal reference line at -6 dB where zero-phase bands meet

        plt.title("3-Band zero-Phase butterworth crossover response")  # Sets plot title
        plt.xlabel("Frequency (Hz)")  # Labels the X axis
        plt.ylabel("Magnitude (dB)")  # Labels the Y axis
        plt.xlim(20, 20000)  # Limits the view to the human audible frequency range (20 Hz to 20 kHz)
        plt.ylim(-60, 6)  # Limits the vertical range from -60 dB up to +6 dB
        plt.grid(True, which="both", linestyle="--", alpha=0.4)  # Shows both major and minor logarithmic grid lines
        plt.legend(loc="lower left")  # Places the legend box in the lower-left corner
        plt.tight_layout()  # Adjusts margins automatically so labels do not get cut off

        if save_path:  # Saves the figure to disk before plt.show() so the image is never saved blank
            plt.savefig(save_path, dpi=150)  # Exports a sharp 150 DPI PNG image
            print(f"Saved filter response plot to '{save_path}'")

        plt.show()  # Displays the plot window on screen

    def plot_spectrum_comparison(self, save_path: str | None = "spectrum_comparison.png"):
        """Compares the FFT frequency spectrum before and after EQ processing."""
        if self.audio_data is None or self.processed_audio is None:  # Guard clause: needs both original and equalized signals
            print("Need both original and processed audio. Run load_audio() and apply_eq() first.")
            return

        # If audio is stereo (ndim > 1), averages left and right channels into mono solely for FFT visualization (original arrays stay untouched)
        orig = np.mean(self.audio_data, axis=1) if self.audio_data.ndim > 1 else self.audio_data
        proc = np.mean(self.processed_audio, axis=1) if self.processed_audio.ndim > 1 else self.processed_audio

        n = len(orig)  # Total number of samples in the signal
        window = np.hanning(n)  # Creates a bell-shaped Hann window that tapers the edges to zero to minimize spectral leakage
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)  # Computes the frequency bin values in Hz for the positive half of the spectrum (0 Hz to nyquist)

        # Normalization factor for a single-sided windowed FFT: divides by window sum and multiplies by 2 so sine amplitudes match true dB FS
        scale = np.sum(window) / 2.0
        orig_mag = np.abs(np.fft.rfft(orig * window)) / scale  # Applies window, runs real FFT on original audio, extracts magnitude, and normalizes
        proc_mag = np.abs(np.fft.rfft(proc * window)) / scale  # Same FFT calculation for the equalized audio

        # Converts linear FFT magnitudes into dB FS with a -200 dB floor (1e-10) to avoid log10(0)
        orig_db = 20 * np.log10(np.maximum(orig_mag, 1e-10))
        proc_db = 20 * np.log10(np.maximum(proc_mag, 1e-10))

        plt.figure(figsize=(10, 5))  # Creates a 10x5 inch figure
        plt.semilogx(freqs, orig_db, label="Original signal", alpha=0.75, linewidth=1.5)  # Plots the original spectrum on a log frequency axis
        plt.semilogx(freqs, proc_db, label="Equalized signal", alpha=0.85, linewidth=1.5)  # Overlays the equalized spectrum for direct comparison

        if self.low_cutoff and self.high_cutoff:  # Draws vertical dotted lines at the crossover frequencies if they are set (non-zero)
            plt.axvline(self.low_cutoff, color="gray", linestyle=":", alpha=0.6)
            plt.axvline(self.high_cutoff, color="gray", linestyle=":", alpha=0.6)

        title = "Frequency Spectrum comparison (FFT): Original vs. Equalized"  # Base title for the FFT plot
        if self.last_gains_db:  # Appends the last applied dB gains on a second line in the title if available
            l_db, m_db, h_db = self.last_gains_db  # Unpacks the saved tuple into three separate variables
            title += f"\n(Low: {l_db:+.1f} dB, Mid: {m_db:+.1f} dB, High: {h_db:+.1f} dB)"

        plt.title(title)  # Applies the dynamic title
        plt.xlabel("Frequency (Hz)")  # Labels the X axis
        plt.ylabel("Amplitude (dB FS)")  # Labels the Y axis (decibels relative to full scale)
        plt.xlim(20, 20000)  # Frames the plot across the audible range (20 Hz to 20 kHz)
        plt.ylim(-80, 6)  # Frames amplitude from -80 dB FS up to +6 dB FS
        plt.grid(True, which="both", linestyle="--", alpha=0.4)  # Adds major and minor grid lines
        plt.legend(loc="upper right")  # Places the legend box in the upper-right corner
        plt.tight_layout()  # Prevents labels or titles from overlapping or clipping

        if save_path:  # Saves the comparison plot as a PNG file before displaying it
            plt.savefig(save_path, dpi=150)
            print(f"Saved spectrum comparison plot to '{save_path}'")

        plt.show()  # Displays the plot window on screen


if __name__ == "__main__":  # Runs this test block only when equalizer.py is executed directly (not when imported by another script)
    eq = ThreeBandEqualizer("test_signal.wav")  # Creates an equalizer instance pointing to the synthetic test audio
    if eq.load_audio():  # Proceeds only if the WAV file loaded successfully
        eq.inspect_signal()  # Prints sample rate, nyquist limit, channels, duration, and peak
        eq.split_bands(low_cutoff=250.0, high_cutoff=3000.0, order=4)  # Splits audio into low, mid, and high bands

        # Test 1: Flat response check (all bands at 0 dB should reconstruct the original signal)
        eq.apply_eq(low_db=0.0, mid_db=0.0, high_db=0.0)

        # Test 2: Actual EQ curve (boost lows, heavily cut mids, keep highs neutral)
        eq.apply_eq(low_db=12.0, mid_db=-24.0, high_db=0.0)
        eq.export_audio("output_eq_test.wav")  # Exports the equalized mix to disk
        eq.plot_filter_responses()  # Plots and saves the 3-band crossover response curve
        eq.plot_spectrum_comparison()  # Plots and saves the original vs. equalized FFT spectrum