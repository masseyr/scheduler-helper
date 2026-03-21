"""
NASA Size Estimation Model (SEM) for orbital debris radar cross-section estimation.

Models debris as a perfectly conducting sphere and applies Mie scattering theory
to compute RCS at any radar frequency, following the methodology used in NASA
characterization studies (e.g., Stansbery et al., NASA TM-4699).

Key functions:
    avgrcs(radius_m, freq_hz)          -- RCS for sphere of given radius at frequency
    translate_rcs(rcs, f_in, f_out)    -- scale RCS between radar bands via size inversion
    estimate_radius(rcs_m2, freq_hz)   -- invert avgrcs to recover sphere radius
    radar_band_center(band)            -- center frequency (Hz) for a named radar band
"""

import math
import cmath

# Speed of light (m/s)
_C = 299792458.0

# Radar band center frequencies in Hz.  Values chosen to match typical
# space-surveillance radars (Haystack/HAX, Millstone, etc.).
RADAR_BANDS = {
    'UHF': 425.0e6,    # HAX / ALTAIR / FPS-85 neighbourhood
    'L':   1.3e9,      # Millstone L-band
    'S':   3.0e9,
    'C':   5.5e9,
    'X':   10.0e9,     # Haystack
}

# Aliases matching the C++ RADARBandEnum names used in mk_satcat_file
_BAND_ALIASES = {
    'UHF_BAND': 'UHF',
    'L_BAND':   'L',
    'S_BAND':   'S',
    'C_BAND':   'C',
    'X_BAND':   'X',
}


def radar_band_center(band: str) -> float:
    """Return centre frequency (Hz) for *band*.

    Accepts short names ('UHF', 'L', 'S', 'C', 'X') and the C++ enum-style
    names ('UHF_BAND', 'L_BAND', …).

    Raises KeyError for unknown bands.
    """
    key = _BAND_ALIASES.get(band, band)
    return RADAR_BANDS[key]


# ---------------------------------------------------------------------------
# Mie scattering for a perfectly conducting (PEC) sphere
# ---------------------------------------------------------------------------

def _mie_pec_backscatter(radius_m: float, freq_hz: float) -> float:
    """Exact backscatter RCS (m²) of a PEC sphere via Mie series.

    The series is truncated using the Wiscombe stopping criterion:
        n_stop = round(x + 4*x^(1/3) + 2), minimum 5
    where x = 2πa/λ is the size parameter.

    Riccati-Bessel functions ψ_n(x) and ξ_n(x) = x h_n^(1)(x) are computed
    by upward recurrence; their derivatives follow from the recurrence identity
        f'_n(x) = f_{n-1}(x) − (n/x) f_n(x).

    Backscatter amplitude:
        S_back = Σ_{n=1}^N  (−1)^(n+1) (2n+1)/2 (a_n − b_n)
    where a_n = ψ_n/ξ_n  (TM/electric),  b_n = ψ'_n/ξ'_n  (TE/magnetic).

    σ_bs = (λ²/π) |S_back|²
    """
    wavelength = _C / freq_hz
    x = 2.0 * math.pi * radius_m / wavelength  # size parameter

    if x < 1e-10:
        return 0.0

    # Stopping criterion (Wiscombe 1980)
    n_stop = int(x + 4.0 * x ** (1.0 / 3.0) + 2.0) + 1
    n_stop = max(n_stop, 5)

    # -----------------------------------------------------------------------
    # Upward recurrence for ψ_n(x) and ξ_n(x)
    # Seed values:
    #   ψ_{−1}(x) = cos x          ξ_{−1}(x) = cos x + i sin x = e^{ix}
    #   ψ_0(x)    = sin x          ξ_0(x)    = sin x − i cos x
    # Recurrence: f_{n+1} = (2n+1)/x · f_n − f_{n−1}
    # -----------------------------------------------------------------------
    psi_prev = math.cos(x)
    psi_curr = math.sin(x)
    xi_prev  = complex(math.cos(x),  math.sin(x))   # e^{ix}
    xi_curr  = complex(math.sin(x), -math.cos(x))

    s_back = 0j

    for n in range(1, n_stop + 1):
        factor = (2 * n - 1) / x

        psi_next = factor * psi_curr - psi_prev
        xi_next  = factor * xi_curr  - xi_prev

        # Mie coefficients for PEC sphere
        an = psi_next / xi_next

        dpsi = psi_curr - (n / x) * psi_next   # ψ'_n(x)
        dxi  = xi_curr  - (n / x) * xi_next    # ξ'_n(x)
        bn   = dpsi / dxi

        s_back += ((-1) ** (n + 1)) * (2 * n + 1) / 2.0 * (an - bn)

        # Advance recurrence
        psi_prev, psi_curr = psi_curr, psi_next
        xi_prev,  xi_curr  = xi_curr,  xi_next

    sigma = (wavelength ** 2 / math.pi) * abs(s_back) ** 2
    return sigma


def avgrcs(radius_m: float, freq_hz: float) -> float:
    """Average RCS (m²) for a sphere of *radius_m* metres at *freq_hz* Hz.

    Uses the exact Mie series solution for a perfectly conducting sphere.
    For a sphere the backscatter equals the orientation-averaged RCS, making
    this directly applicable to tumbling debris.

    Args:
        radius_m: Sphere radius in metres (half the physical diameter).
        freq_hz:  Radar centre frequency in Hz.

    Returns:
        RCS in m².  Returns 0.0 for radius ≤ 0.
    """
    if radius_m <= 0.0:
        return 0.0
    return _mie_pec_backscatter(radius_m, freq_hz)


def estimate_radius(rcs_m2: float, freq_hz: float,
                    r_min: float = 1e-4, r_max: float = 1e4,
                    tol: float = 1e-6) -> float:
    """Invert *avgrcs* to recover the sphere radius that produces *rcs_m2*.

    Uses bisection in log-radius space.  The Mie series is not strictly
    monotone in the resonance region, so this returns the *smallest* radius
    consistent with the measured RCS; this is appropriate for SSA use because
    the probability distribution of debris sizes favours smaller objects.

    Args:
        rcs_m2:  Target RCS in m².
        freq_hz: Radar frequency in Hz.
        r_min:   Lower bound for search (metres).  Default 0.1 mm.
        r_max:   Upper bound for search (metres).  Default 10 km.
        tol:     Convergence tolerance on radius (metres).

    Returns:
        Estimated radius in metres, or -1.0 if no solution found in [r_min, r_max].
    """
    if rcs_m2 <= 0.0:
        return -1.0

    f_lo = avgrcs(r_min, freq_hz) - rcs_m2
    f_hi = avgrcs(r_max, freq_hz) - rcs_m2

    if f_lo > 0.0:
        return r_min   # RCS already exceeded at minimum radius
    if f_hi < 0.0:
        return -1.0    # No solution within bounds

    # Bisection in log space for better convergence across many decades
    log_lo = math.log(r_min)
    log_hi = math.log(r_max)

    for _ in range(100):
        log_mid = 0.5 * (log_lo + log_hi)
        r_mid = math.exp(log_mid)
        if abs(log_hi - log_lo) * r_mid < tol:
            break
        f_mid = avgrcs(r_mid, freq_hz) - rcs_m2
        if f_mid <= 0.0:
            log_lo = log_mid
        else:
            log_hi = log_mid

    return math.exp(0.5 * (log_lo + log_hi))


def translate_rcs(rcs_in: float, freq_in_hz: float, freq_out_hz: float) -> float:
    """Translate a measured RCS from one radar frequency to another.

    Inverts *avgrcs* at *freq_in_hz* to obtain an equivalent sphere radius,
    then evaluates *avgrcs* at *freq_out_hz*.  This is the same two-step
    approach used in the C++ translate_rcs() implementation.

    Args:
        rcs_in:      Known RCS in m².
        freq_in_hz:  Frequency at which rcs_in was measured (Hz).
        freq_out_hz: Target frequency (Hz).

    Returns:
        Translated RCS in m², or -1.0 if size inversion fails.
    """
    r = estimate_radius(rcs_in, freq_in_hz)
    if r < 0.0:
        return -1.0
    return avgrcs(r, freq_out_hz)
