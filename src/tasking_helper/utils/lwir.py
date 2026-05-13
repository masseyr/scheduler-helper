"""
lwir.py -- Simplified LWIR irradiance model for RSO thermal detection.

Implements the emitted-energy model described in the SSNACS IR sensor
documentation.  Only thermal emission is modelled; reflected solar
contribution is NOT included.  This is appropriate for LWIR (~= 8-14 um)
where emission dominates.  MWIR (3-5 um) receives significant reflected
solar flux and requires a separate algorithm.

Physics summary
---------------
Spectral radiance (Planck's law, modified for gray bodies):

    B_lam(lam, T) = 2hepsc^2 / lam^5  *  1 / (exp(hc/(lamkT)) - 1)

    [W / (m^2 sr um)],  lam in um

In-band radiance (integrated over the sensor band [lam_1, lam_2]):

    L = integral B_lam(lam, T) dlam      [W / (m^2 sr)]

Solid angle subtended by a spherical RSO (radius R at range d, R << d):

    Omega = pi (R / d)^2           [sr]

Irradiance at the sensor aperture:

    E = alpha * L * Omega            [W / m^2]

where alpha in [0, 1] is the broadband atmospheric transmission.

Caveats
-------
* Emitted energy only.  MWIR bands require a different model.
* RSO modelled as a spherical gray body -- no geometry, gradients, or
  material variability.
* alpha is a constant scale factor; wavelength-dependent transmission is
  not modelled.
* RSO temperatures (sunlit / eclipse) are analyst-supplied; thermal
  equilibrium is not computed.

Usage
-----
    from tasking_helper.utils.lwir import LWIRSensor

    sensor = LWIRSensor(
        lambda1_um=8.0, lambda2_um=12.0,
        irradiance_cutoff_W_m2=1e-11,
        atm_absorption=0.85,
        T_sunlit_K=350.0,
        T_eclipse_K=240.0,
        emissivity=0.90,
    )

    result = sensor.detect(R_m=1.5, d_m=500e3, in_eclipse=False)
    print(result.detected, result.irradiance_W_m2)

    # Maximum detection range for a given RSO
    d_max = sensor.max_range(R_m=1.5, in_eclipse=False)
    print(f"Max range: {d_max/1e3:.1f} km")


Other notes
-----------
    Stefan-Boltzmann check: 0.065% error -- the Planck formula integrates to sigmaT^4/pi as expected
    300K object at 500 km, R=1.5m: E = 1.69x10^-^9 W/m^2 -> 22 dB above cutoff
    Eclipse (240K) detection fails at 5,000 km -- colder target, smaller solid angle, dropped below cutoff
    Max range scales linearly with R (as expected from Omega proportional_to R^2)

    LWIRSensor class:
        Configures band limits, irradiance cutoff, alpha, T_sunlit, T_eclipse, eps
        Pre-computes and caches in-band radiance on init (and via precompute())
        detect(R_m, d_m, in_eclipse) -> DetectionResult with irradiance, solid angle, margin_dB
        max_range(R_m, in_eclipse) -> maximum detection distance [m]
    
    Caveats (from document):
        Emitted energy only -- MWIR needs a different model (reflected solar not included)
        alpha is a constant broadband factor -- no wavelength-dependent transmission


Dependencies: numpy, scipy
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from scipy.integrate import quad

__all__ = [
    "planck_spectral_radiance",
    "in_band_radiance",
    "solid_angle_sphere",
    "compute_irradiance",
    "LWIRSensor",
    "DetectionResult",
    "LWIR_8_12",
    "LWIR_8_14",
]

# -- Physical constants (SI) ---------------------------------------------------

_H  = 6.626_070_15e-34    # Planck constant      [J*s]
_C  = 2.997_924_58e8      # speed of light       [m/s]
_KB = 1.380_649e-23       # Boltzmann constant   [J/K]

# "um-friendly" radiation constants so lam can be supplied directly in um:
#   C1  [W*um^4/(m^2*sr)] = 2hc^2 x 10^2^4
#   C2  [um*K]           = hc/k x 10^6
_C1 = 2.0 * _H * _C ** 2 * 1e24   # ~= 1.19104 x 10^8
_C2 = (_H * _C / _KB) * 1e6       # ~= 14387.8

# -- Named band limits [um] ----------------------------------------------------

LWIR_8_12 = (8.0, 12.0)   # standard LWIR band
LWIR_8_14 = (8.0, 14.0)   # extended LWIR band


# -- Physical functions --------------------------------------------------------

def planck_spectral_radiance(lam_um: float,
                              T_K:    float,
                              emissivity: float = 1.0) -> float:
    """
    Spectral radiance of a gray body at wavelength lam_um and temperature T_K.

    Parameters
    ----------
    lam_um    : wavelength [um]
    T_K       : temperature [K]
    emissivity: eps in (0, 1]  (default 1 = perfect blackbody)

    Returns
    -------
    B_lam : float  [W / (m^2 sr um)]

    Formula
    -------
    B_lam = eps * C1 / (lam^5 * (exp(C2 / (lamT)) - 1))
    with C1 = 2hc^2 x 10^2^4 [W*um^4/(m^2*sr)],  C2 = hc/k x 10^6 [um*K].
    """
    return emissivity * _C1 / (lam_um ** 5 * (math.exp(_C2 / (lam_um * T_K)) - 1.0))


def in_band_radiance(T_K:       float,
                     emissivity: float = 1.0,
                     lam1_um:   float = 8.0,
                     lam2_um:   float = 12.0) -> float:
    """
    In-band radiance -- numerically integrate B_lam over [lam1_um, lam2_um].

    Parameters
    ----------
    T_K       : temperature [K]
    emissivity: eps in (0, 1]
    lam1_um   : lower wavelength bound [um]
    lam2_um   : upper wavelength bound [um]

    Returns
    -------
    L : float  [W / (m^2 sr)]
    """
    L, _err = quad(
        planck_spectral_radiance,
        lam1_um, lam2_um,
        args=(T_K, emissivity),
        limit=200, epsabs=1e-8, epsrel=1e-10,
    )
    return L


def solid_angle_sphere(R_m: float, d_m: float) -> float:
    """
    Solid angle subtended by a spherical RSO of radius R at range d (R << d).

    Parameters
    ----------
    R_m : effective sphere radius [m]
    d_m : range from observer to RSO [m]

    Returns
    -------
    Omega : float  [sr]

    Formula
    -------
    Omega = pi (R / d)^2
    """
    return math.pi * (R_m / d_m) ** 2


def compute_irradiance(L:              float,
                        R_m:           float,
                        d_m:           float,
                        atm_absorption: float = 1.0) -> float:
    """
    Irradiance at a sensor from an RSO.

    Parameters
    ----------
    L              : in-band radiance of RSO [W / (m^2 sr)]
    R_m            : RSO effective radius [m]
    d_m            : sensor-to-RSO range [m]
    atm_absorption : broadband atmospheric transmission alpha in [0, 1]

    Returns
    -------
    E : float  [W / m^2]

    Formula
    -------
    E = alpha * L * Omega   where Omega = pi(R/d)^2
    """
    return atm_absorption * L * solid_angle_sphere(R_m, d_m)


# -- Result type ---------------------------------------------------------------

class DetectionResult(NamedTuple):
    """Structured result from LWIRSensor.detect().

    Attributes
    ----------
    detected              : True if E >= irradiance_cutoff
    irradiance_W_m2       : computed irradiance E at sensor aperture [W/m^2]
    in_band_radiance_W_m2_sr : in-band radiance L of the RSO [W/(m^2*sr)]
    solid_angle_sr        : solid angle Omega subtended by the RSO [sr]
    T_K                   : RSO temperature used [K]
    emissivity            : emissivity used
    in_eclipse            : whether the eclipse temperature was applied
    margin_dB             : 10*log10(E / cutoff)  -- positive means detected
    """
    detected:                 bool
    irradiance_W_m2:          float
    in_band_radiance_W_m2_sr: float
    solid_angle_sr:           float
    T_K:                      float
    emissivity:               float
    in_eclipse:               bool
    margin_dB:                float


# -- Sensor class --------------------------------------------------------------

class LWIRSensor:
    """
    Simplified LWIR irradiance sensor model.

    Models thermal detection of an RSO based on emitted in-band radiance,
    range-dependent solid angle, and atmospheric attenuation.  A pre-computed
    radiance cache is used to speed up repeated calls at the same temperature
    and emissivity.

    Parameters
    ----------
    lambda1_um             : lower wavelength band limit [um]  (default 8.0)
    lambda2_um             : upper wavelength band limit [um]  (default 12.0)
    irradiance_cutoff_W_m2 : minimum detectable irradiance E_min [W/m^2]
    atm_absorption         : broadband atmospheric transmission alpha in [0, 1]
    T_sunlit_K             : assumed RSO temperature when sunlit [K]
    T_eclipse_K            : assumed RSO temperature when in eclipse [K]
    emissivity             : default RSO gray-body emissivity eps in (0, 1]
    """

    def __init__(self,
                 lambda1_um:             float = 8.0,
                 lambda2_um:             float = 12.0,
                 irradiance_cutoff_W_m2: float = 1e-11,
                 atm_absorption:         float = 0.9,
                 T_sunlit_K:             float = 350.0,
                 T_eclipse_K:            float = 250.0,
                 emissivity:             float = 0.90) -> None:
        if not (0.0 < emissivity <= 1.0):
            raise ValueError(f"emissivity must be in (0, 1], got {emissivity}")
        if not (0.0 <= atm_absorption <= 1.0):
            raise ValueError(f"atm_absorption must be in [0, 1], got {atm_absorption}")
        if lambda1_um >= lambda2_um:
            raise ValueError("lambda1_um must be less than lambda2_um")
        if irradiance_cutoff_W_m2 <= 0.0:
            raise ValueError("irradiance_cutoff_W_m2 must be positive")

        self.lambda1_um             = lambda1_um
        self.lambda2_um             = lambda2_um
        self.irradiance_cutoff_W_m2 = irradiance_cutoff_W_m2
        self.atm_absorption         = atm_absorption
        self.T_sunlit_K             = T_sunlit_K
        self.T_eclipse_K            = T_eclipse_K
        self.emissivity             = emissivity
        self._cache: dict           = {}

        # Pre-compute for default temperatures
        self.precompute([self.T_sunlit_K, self.T_eclipse_K])

    # -- Radiance cache ---------------------------------------------------------

    def precompute(self,
                   temperatures_K: list[float],
                   emissivity:      float | None = None) -> None:
        """
        Pre-compute and cache in-band radiance for a list of temperatures.

        Call this before tight loops to avoid recomputing the Planck integral.

        Parameters
        ----------
        temperatures_K : list of temperatures [K] to cache
        emissivity     : emissivity to use (default: sensor default)
        """
        eps = emissivity if emissivity is not None else self.emissivity
        for T in temperatures_K:
            key = (round(T, 4), round(eps, 6),
                   self.lambda1_um, self.lambda2_um)
            if key not in self._cache:
                self._cache[key] = in_band_radiance(
                    T, eps, self.lambda1_um, self.lambda2_um)

    def _get_radiance(self, T_K: float, emissivity: float) -> float:
        key = (round(T_K, 4), round(emissivity, 6),
               self.lambda1_um, self.lambda2_um)
        if key not in self._cache:
            self._cache[key] = in_band_radiance(
                T_K, emissivity, self.lambda1_um, self.lambda2_um)
        return self._cache[key]

    # -- Public API -------------------------------------------------------------

    def get_in_band_radiance(self,
                              T_K:       float,
                              emissivity: float | None = None) -> float:
        """
        In-band radiance L of an RSO at temperature T_K [W/(m^2*sr)].

        Uses the internal cache; calls scipy.integrate.quad on first access.
        """
        return self._get_radiance(T_K, emissivity if emissivity is not None
                                        else self.emissivity)

    def get_irradiance(self,
                        T_K:       float,
                        R_m:       float,
                        d_m:       float,
                        emissivity: float | None = None) -> float:
        """
        Irradiance E at the sensor aperture from an RSO [W/m^2].

        Parameters
        ----------
        T_K       : RSO temperature [K]
        R_m       : RSO effective radius [m]
        d_m       : sensor-to-RSO range [m]
        emissivity: eps (default: sensor default)

        Returns
        -------
        E : float  [W/m^2]
        """
        eps = emissivity if emissivity is not None else self.emissivity
        L   = self._get_radiance(T_K, eps)
        return compute_irradiance(L, R_m, d_m, self.atm_absorption)

    def detect(self,
               R_m:        float,
               d_m:        float,
               in_eclipse:  bool        = False,
               T_K:         float | None = None,
               emissivity:  float | None = None) -> DetectionResult:
        """
        Evaluate whether the sensor can detect an RSO.

        Parameters
        ----------
        R_m       : RSO effective radius [m]
        d_m       : sensor-to-RSO range [m]
        in_eclipse: use T_eclipse_K if True, T_sunlit_K if False
        T_K       : override temperature [K]  (ignores in_eclipse if set)
        emissivity: override emissivity (default: sensor default)

        Returns
        -------
        DetectionResult
        """
        if T_K is None:
            T_K = self.T_eclipse_K if in_eclipse else self.T_sunlit_K
        eps   = emissivity if emissivity is not None else self.emissivity
        L     = self._get_radiance(T_K, eps)
        Omega = solid_angle_sphere(R_m, d_m)
        E     = self.atm_absorption * L * Omega
        ratio = E / self.irradiance_cutoff_W_m2 if self.irradiance_cutoff_W_m2 > 0 else 0.0
        margin = 10.0 * math.log10(ratio) if ratio > 0 else -math.inf
        return DetectionResult(
            detected                 = E >= self.irradiance_cutoff_W_m2,
            irradiance_W_m2          = E,
            in_band_radiance_W_m2_sr = L,
            solid_angle_sr           = Omega,
            T_K                      = T_K,
            emissivity               = eps,
            in_eclipse               = in_eclipse,
            margin_dB                = margin,
        )

    def max_range(self,
                  R_m:       float,
                  in_eclipse: bool        = False,
                  T_K:        float | None = None,
                  emissivity: float | None = None) -> float:
        """
        Maximum detection range for an RSO of radius R_m [m].

        Solves  alpha * L * pi(R/d_max)^2 = cutoff  for d_max.

        Parameters
        ----------
        R_m       : RSO effective radius [m]
        in_eclipse: use eclipse temperature if True
        T_K       : override temperature [K]
        emissivity: override emissivity

        Returns
        -------
        d_max : float  [m]
        """
        if T_K is None:
            T_K = self.T_eclipse_K if in_eclipse else self.T_sunlit_K
        eps = emissivity if emissivity is not None else self.emissivity
        L   = self._get_radiance(T_K, eps)
        # d_max = R * sqrt(alpha * L * pi / cutoff)
        return R_m * math.sqrt(
            self.atm_absorption * L * math.pi / self.irradiance_cutoff_W_m2)

    def irradiance_table(self,
                          R_m:        float,
                          d_m_values: list[float],
                          in_eclipse:  bool = False,
                          T_K:         float | None = None,
                          emissivity:  float | None = None,
                         ) -> list[tuple[float, float, bool]]:
        """
        Compute irradiance over a range of distances.

        Returns list of (d_m, E [W/m^2], detected).
        """
        return [(d, self.get_irradiance(
                        T_K if T_K is not None else
                        (self.T_eclipse_K if in_eclipse else self.T_sunlit_K),
                        R_m, d, emissivity),
                 self.get_irradiance(
                        T_K if T_K is not None else
                        (self.T_eclipse_K if in_eclipse else self.T_sunlit_K),
                        R_m, d, emissivity) >= self.irradiance_cutoff_W_m2)
                for d in d_m_values]

    # -- Display helpers --------------------------------------------------------

    def __str__(self) -> str:
        return (
            f"LWIRSensor("
            f"band={self.lambda1_um}-{self.lambda2_um} um, "
            f"cutoff={self.irradiance_cutoff_W_m2:.2e} W/m2, "
            f"alpha={self.atm_absorption:.2f}, "
            f"T_sun={self.T_sunlit_K} K, "
            f"T_ecl={self.T_eclipse_K} K, "
            f"eps={self.emissivity:.2f})"
        )

    def summary(self) -> str:
        """Return a multi-line parameter and pre-computed radiance summary."""
        L_sun = self._get_radiance(self.T_sunlit_K, self.emissivity)
        L_ecl = self._get_radiance(self.T_eclipse_K, self.emissivity)
        return (
            f"LWIR Sensor Summary\n"
            f"  Band            : {self.lambda1_um}-{self.lambda2_um} um\n"
            f"  Cutoff          : {self.irradiance_cutoff_W_m2:.3e} W/m2\n"
            f"  Atm absorption  : {self.atm_absorption:.3f}\n"
            f"  Emissivity      : {self.emissivity:.3f}\n"
            f"  T sunlit        : {self.T_sunlit_K:.1f} K  "
            f"-> L = {L_sun:.4f} W/(m2 sr)\n"
            f"  T eclipse       : {self.T_eclipse_K:.1f} K  "
            f"-> L = {L_ecl:.4f} W/(m2 sr)"
        )
