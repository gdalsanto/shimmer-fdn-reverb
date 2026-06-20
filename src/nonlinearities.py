import numpy as np


## ---------- UTILITIES ----------


class DCBlocker:
    """First-order DC blocker with optional slow energy compensation."""

    def __init__(
        self,
        R: float = 0.995,
        correct_loss: bool = False,
        fs: float = 48000.0,
        env_tau_s: float = 0.05,  # RMS tracking time constant (50 ms)
        gain_tau_s: float = 0.02,  # gain smoothing time constant (20 ms)
    ):
        self.R = float(R)
        self.prev_x = 0.0
        self.prev_y = 0.0

        self.correct_loss = bool(correct_loss)

        # Envelope follower states (power domain)
        self.eps = 1e-12
        self.in_pow = 1e-12
        self.out_pow = 1e-12
        self.gain = 1.0

        self.alpha_env = np.exp(-1.0 / (fs * env_tau_s))
        self.alpha_gain = np.exp(-1.0 / (fs * gain_tau_s))

    def __call__(self, x):
        # Scalar-safe conversion
        x = float(np.asarray(x).item())

        # DC blocker
        y = x - self.prev_x + self.R * self.prev_y
        self.prev_x = x
        self.prev_y = y

        if self.correct_loss:
            # Smooth input/output power
            self.in_pow = self.alpha_env * self.in_pow + (1.0 - self.alpha_env) * (
                x * x
            )
            self.out_pow = self.alpha_env * self.out_pow + (1.0 - self.alpha_env) * (
                y * y
            )

            # Target RMS correction
            target_gain = np.sqrt((self.in_pow + self.eps) / (self.out_pow + self.eps))

            # Smooth and apply gain
            self.gain = (
                self.alpha_gain * self.gain + (1.0 - self.alpha_gain) * target_gain
            )
            y *= self.gain

        return y

    def reset_states(self):
        self.prev_x = 0.0
        self.prev_y = 0.0
        self.in_pow = 1e-12
        self.out_pow = 1e-12
        self.gain = 1.0


class RampFade:
    """Linear ramp fade."""

    def __init__(self, fade_length_samps: int):
        self.fade_length = fade_length_samps
        self.sample_count = 0

    def __call__(self, y, x):
        if self.sample_count < self.fade_length:
            fade_value = self.sample_count / self.fade_length
            g = np.sqrt(2 - 2*np.abs(fade_value - 0.5)) 
            self.sample_count += 1
            return g * (y * fade_value + x * (1 - fade_value))
        else:
            return y


def get_nl_function(
    nl_type: str,
    N: int,
    positions: list = None,
    extra_arg=None,
    add_dc_block: bool = False,
):
    # Check if extra_arg contains fade specifications and convert to callable if needed
    try:
        if extra_arg["fade"] == "ramp":
            extra_arg["fade"] = RampFade(extra_arg["fade_length_samps"])
        elif extra_arg["fade"] == "const":
            alpha = extra_arg["fade_const"]
            extra_arg["fade"] = lambda y, x: np.sqrt(2 - 2*np.abs(alpha - 0.5)) * (y * alpha + x * (1 - alpha))
    except Exception:
        # create the extra_arg if it doesn't exist and set fade to identity
        if extra_arg is None:
            extra_arg = {}
        extra_arg["fade"] = lambda y, x: y

    # Check if DC blocker is requested and initialize nl_matrix accordingly
    if add_dc_block:
        nl_matrix = [[lambda x: x for _ in range(N)], [lambda x: x for _ in range(N)]]
    else:
        nl_matrix = [lambda x: x for _ in range(N)]

    # Define the nonlinearity function based on nl_type and populate nl_matrix at specified positions
    for pos in positions or range(N):
        if nl_type == "cfwr":
            # antiderivative of absolute value with optional fading
            cur_nl = cfwr(extra_arg["degree"], alpha=extra_arg["fade"])

        elif nl_type == "granular":
            # Granular pitch shifter
            cur_nl = GranularDelay(
                extra_arg["max_delay_samps"],
                extra_arg["grain_dur_samps"],
                transpose_cents=extra_arg.get("transpose_cents", None),
                fade_ratio=extra_arg.get("fade_ratio", 0.25),
                seed=extra_arg.get("seed", None),
            )

        elif nl_type == "pitchshift":
            # Dual-read-head pitch shifter
            cur_nl = PitchShift(
                extra_arg["max_delay_samps"],
                extra_arg["window_size"],
                extra_arg.get("transpose_cents", 0),
                extra_arg.get("fs", 48000),
                min_delay_samps=extra_arg.get("min_delay_samps", 2),
            )

        elif nl_type == "sdfd":
            # Signal-Dependent Fractional Delay filter
            cur_nl = SDFD(extra_arg["d"])

        else:
            raise ValueError(f"Nonlinearity type '{nl_type}' not recognized.")

        if add_dc_block:
            nl_matrix[0][pos] = cur_nl
        else:
            nl_matrix[pos] = cur_nl

    # Add DC blockers if requested
    if add_dc_block:
        for pos in positions or range(N):
            nl_matrix[1][pos] = DCBlocker(R=0.995, correct_loss=True)

    return nl_matrix


## ---------- NON LINEARITIES ----------


class cfwr:
    """Antiderivative of absolute value nonlinearity, with optional output fading."""

    def __init__(self, degree: int = 1, alpha: callable = lambda y, x: y):
        self.degree = degree
        self.alpha = alpha
        self.state = np.zeros(self.degree)

    def anti_dev_1(self, x):
        y = 0.5 * x * np.abs(x)
        return y

    def anti_dev_2(self, x):
        y = (1 / 6) * np.abs(x) ** 3  # + x ( we assumed C = 0)
        return y

    def __call__(self, x):
        if self.degree == 0:
            y = np.abs(x)
            return y

        elif self.degree == 1:
            y = self.anti_dev_1(x) - self.anti_dev_1(self.state[-1])
            den = x - self.state
            if np.abs(den) <= 1e-8:
                y = np.abs(x + self.state[-1]) / 2 
            else:
                y = y / den
            self.state[-1] = x.item()

        elif self.degree == 2:
            y_1 = self.anti_dev_2(x) - self.anti_dev_2(self.state[-1])

            den_1 = x - self.state[-1]
            if den_1 == 0:
                den_1 = 1e-12
            y_1 = y_1 / den_1

            y_2 = self.anti_dev_2(self.state[-1]) - self.anti_dev_2(self.state[-2])

            den_2 = self.state[-1] - self.state[-2]
            if den_2 == 0:
                den_2 = 1e-12
            y_2 = y_2 / den_2

            den_3 = x - self.state[-2]
            if den_3 == 0:
                den_3 = 1e-12
            y = 2 / den_3 * (y_1 - y_2)
            self.state = np.roll(self.state, -1)
            self.state[-1] = x.item()

        y = self.alpha(y, x)

        return y


class SDFD:
    """Signal-Dependent Fractional Delay (SDFD) filter."""

    def __init__(self, d: float):
        self.d = d

        # Delay states (p: positive, n: negative)
        self.p1 = 0.0  # p(n-1)
        self.p2 = 0.0  # p(n-2)
        self.n1 = 0.0  # n(n-1)

    @staticmethod
    def pos_half_wave(x):
        return max(x, 0.0)

    @staticmethod
    def neg_half_wave(x):
        return min(x, 0.0)

    def __call__(self, x):
        d = self.d

        # Half-wave rectifiers
        p = self.pos_half_wave(x)
        n = self.neg_half_wave(x)

        # Negative branch sum
        s1 = (1 - d) * self.n1 + d * n

        # Positive branch sum
        y = s1 + d * self.p2 + (1 - d) * self.p1

        # Update delays
        self.p2 = self.p1
        self.p1 = p
        self.n1 = n

        return y

    def reset_states(self):
        self.p1 = 0.0
        self.p2 = 0.0
        self.n1 = 0.0


class PitchShift:
    """Dual-read-head pitch shifter using one circular buffer."""

    def __init__(
        self,
        max_delay_samps: int,
        window_size: int,
        transpose_cents: float = 0.0,
        fs: float = 48000.0,
        min_delay_samps: int = 2,
    ):
        assert max_delay_samps > window_size + min_delay_samps, (
            "max_delay_samps must be > window_size + min_delay_samps"
        )

        self.max_delay = int(max_delay_samps)
        self.window_size = int(window_size)
        self.min_delay = int(min_delay_samps)
        self.fs = float(fs)

        self.buffer = np.zeros(self.max_delay, dtype=float)
        self.write_ptr = 0

        self.phase_1 = 0.0
        self.phase_2 = 0.5  # 180° offset

        self.set_transpose_cents(transpose_cents)

        # Optional logs for analysis/debug
        self.write_ptrs = []
        self.read_ptrs_1 = []
        self.read_ptrs_2 = []
        self.fade_vals_1 = []
        self.fade_vals_2 = []
        self.sig_1 = []
        self.sig_2 = []

    def set_transpose_cents(self, cents: float):
        # True cents -> ratio
        self.transpose_cents = float(cents)
        self.pitch_ratio = 2.0 ** (self.transpose_cents / 1200.0)
        # Delay slope: dD/dn = 1 - ratio, with D = min_delay + phase*window
        self.phase_inc = (1.0 - self.pitch_ratio) / float(self.window_size)

    def reset_states(self):
        self.buffer.fill(0.0)
        self.write_ptr = 0
        self.phase_1 = 0.0
        self.phase_2 = 0.5
        self.write_ptrs.clear()
        self.read_ptrs_1.clear()
        self.read_ptrs_2.clear()
        self.fade_vals_1.clear()
        self.fade_vals_2.clear()
        self.sig_1.clear()
        self.sig_2.clear()

    def _read_interpolated(self, ptr: float) -> float:
        ptr %= self.max_delay
        i = int(ptr)
        f = ptr - i

        y0 = self.buffer[(i - 1) % self.max_delay]
        y1 = self.buffer[i]
        y2 = self.buffer[(i + 1) % self.max_delay]
        y3 = self.buffer[(i + 2) % self.max_delay]

        a = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3
        b = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
        c = -0.5 * y0 + 0.5 * y2
        d = y1
        return a * f**3 + b * f**2 + c * f + d

    def __call__(self, sample):
        x = float(np.asarray(sample).item())

        # Write input
        w = self.write_ptr
        self.buffer[w] = x
        self.write_ptr = (self.write_ptr + 1) % self.max_delay

        # Two delay trajectories/read heads
        d1 = self.min_delay + self.phase_1 * self.window_size
        d2 = self.min_delay + self.phase_2 * self.window_size

        rp1 = w - d1
        rp2 = w - d2

        s1 = self._read_interpolated(rp1)
        s2 = self._read_interpolated(rp2)

        # Half-wave sine equal-power style crossfade
        f1 = np.sin(np.pi * self.phase_1)
        f2 = np.sin(np.pi * self.phase_2)

        y = s1 * f1 + s2 * f2

        # Advance phases
        self.phase_1 = (self.phase_1 + self.phase_inc) % 1.0
        self.phase_2 = (self.phase_2 + self.phase_inc) % 1.0

        # Logs
        self.write_ptrs.append(w)
        self.read_ptrs_1.append(rp1 % self.max_delay)
        self.read_ptrs_2.append(rp2 % self.max_delay)
        self.fade_vals_1.append(f1)
        self.fade_vals_2.append(f2)
        self.sig_1.append(s1)
        self.sig_2.append(s2)

        return np.array(y)


class GranularDelay:
    """Sample-by-sample granular delay processor."""

    def __init__(
        self,
        max_delay_samps: int,
        grain_dur_samps: int,
        transpose_cents: float | None = None,
        fade_ratio: float = 0.25,
        seed: int | None = None,
    ):
        assert max_delay_samps > 2 * grain_dur_samps, (
            "max_delay_samps must be greater than 2 * grain_dur_samps"
        )
        assert 0 < fade_ratio <= 0.5, "fade_ratio must be in (0, 0.5]"

        self.max_delay = max_delay_samps
        self.grain_dur = grain_dur_samps
        self.fade_ratio = float(fade_ratio)
        self.rng = np.random.default_rng(seed)

        # Pitch target
        self.set_transpose_cents(float(transpose_cents))

        # Shared circular write buffer
        self.buffer = np.zeros(max_delay_samps)
        self.write_ptr = 0
        self.samples_written = 0

        # Two grains interleaved with 180° phase offset
        # Grain A starts at sample 0, grain B starts half a grain later
        self.grains = [
            self._new_grain(phase_offset=0),
            self._new_grain(phase_offset=grain_dur_samps // 2),
        ]

        # Logging vectors (one entry per processed sample)
        self.write_ptrs = []  # write pointer position
        self.read_ptrs = [[], []]  # read pointer per grain
        self.fade_vals = [[], []]  # envelope value per grain
        self.grain_sigs = [[], []]  # raw (pre-envelope) grain output

    def set_transpose_cents(self, cents: float):
        # True cents -> ratio
        self.transpose_cents = float(cents)
        self.pitch_ratio = 2.0 ** (self.transpose_cents / 1200.0)
        # Delay slope: dD/dn = 1 - ratio, with D = min_delay + phase*window
        self.delay_inc = 1.0 - self.pitch_ratio

    def _random_read_start(self) -> float:
        """Pick a random starting read position inside the filled buffer."""
        filled = min(self.samples_written, self.max_delay)
        # Keep at least grain_dur samples of look-back so we don't overshoot
        max_age = max(filled - self.grain_dur, 1)
        # age = how many samples behind the write pointer we start reading
        age = self.rng.integers(self.grain_dur, max_age + self.grain_dur)
        # Convert age to an absolute buffer position
        start = (self.write_ptr - int(age)) % self.max_delay
        return float(start)

    def _new_grain(self, phase_offset: int = 0) -> dict:
        return {
            "read_ptr": self._random_read_start(),
            "pos": phase_offset
            % self.grain_dur,  # position within grain (0 … grain_dur-1)
        }

    def _grain_envelope(self, pos: int) -> float:
        fade_len = int(self.fade_ratio * self.grain_dur)
        if fade_len == 0:
            return 1.0
        if pos < fade_len:
            # Fade-in: quarter-wave sine  0 → 1
            return np.sin(0.5 * np.pi * pos / fade_len)
        elif pos >= self.grain_dur - fade_len:
            # Fade-out: quarter-wave cosine  1 → 0
            pos_in_fade = pos - (self.grain_dur - fade_len)
            return np.cos(0.5 * np.pi * pos_in_fade / fade_len)
        else:
            return 1.0

    def _read_interpolated(self, ptr: float) -> float:
        """Cubic interpolated read from the shared buffer."""
        ptr = ptr % self.max_delay
        i = int(ptr)
        f = ptr - i

        y0 = self.buffer[(i - 1) % self.max_delay]
        y1 = self.buffer[i]
        y2 = self.buffer[(i + 1) % self.max_delay]
        y3 = self.buffer[(i + 2) % self.max_delay]

        a = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3
        b = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
        c = -0.5 * y0 + 0.5 * y2
        d = y1
        return a * f**3 + b * f**2 + c * f + d

    def __call__(self, sample) -> np.ndarray:
        x = float(np.asarray(sample).item())

        # Write incoming sample to buffer
        self.buffer[self.write_ptr] = x
        self.write_ptr = (self.write_ptr + 1) % self.max_delay
        self.samples_written += 1

        output = 0.0
        for idx, g in enumerate(self.grains):
            # Read and apply grain envelope
            val = self._read_interpolated(g["read_ptr"])
            env = self._grain_envelope(g["pos"])
            output += val * env

            # Log per-grain state
            self.read_ptrs[idx].append(g["read_ptr"])
            self.fade_vals[idx].append(env)
            self.grain_sigs[idx].append(val)

            # Advance read pointer and grain position
            g["read_ptr"] = (g["read_ptr"] + self.pitch_ratio) % self.max_delay
            g["pos"] += 1

            # Grain finished → restart at a new random position
            if g["pos"] >= self.grain_dur:
                g["read_ptr"] = self._random_read_start()
                g["pos"] = 0

        # Log write pointer
        self.write_ptrs.append(self.write_ptr)

        return np.array(output)

    def reset_states(self):
        """Reset internal state (buffer, pointers, logs)."""
        self.buffer = np.zeros(self.max_delay)
        self.write_ptr = 0
        self.samples_written = 0
        self.grains = [
            self._new_grain(phase_offset=0),
            self._new_grain(phase_offset=self.grain_dur // 2),
        ]
        self.write_ptrs = []
        self.read_ptrs = [[], []]
        self.fade_vals = [[], []]
        self.grain_sigs = [[], []]
