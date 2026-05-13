"""Validation and demonstration of lwir.py."""
import math
import numpy as np
from tasking_helper.utils.lwir import (
    planck_spectral_radiance, in_band_radiance,
    solid_angle_sphere, compute_irradiance,
    LWIRSensor, LWIR_8_12,
)

SEP = "-" * 65

# -- 1. Planck law spot checks --------------------------------------------------
print("=== Planck spectral radiance [W/(m2 sr um)] ===")
print(f"  {'lambda [um]':>12}  {'T [K]':>7}  {'eps':>5}  {'B_lambda':>14}")
print("  " + "-" * 45)
cases = [(10.0, 300, 1.0), (10.0, 300, 0.9), (8.0, 300, 1.0),
         (12.0, 300, 1.0), (10.0, 350, 1.0), (10.0, 250, 1.0)]
for lam, T, eps in cases:
    B = planck_spectral_radiance(lam, T, eps)
    print(f"  {lam:>12.1f}  {T:>7.0f}  {eps:>5.2f}  {B:>14.4f}")

# -- 2. Stefan-Boltzmann consistency check -------------------------------------
# Integral of B_lambda over all wavelengths should equal sigma*T^4 / pi
print()
print("=== Stefan-Boltzmann consistency (eps=1, T=300K) ===")
from scipy.integrate import quad
L_total, _ = quad(planck_spectral_radiance, 0.1, 200.0, args=(300.0, 1.0))
sigma = 5.670374419e-8
SB_expected = sigma * 300.0**4 / math.pi
print(f"  Integrated B_lambda (0.1-200 um): {L_total:.4f} W/(m2 sr)")
print(f"  sigma*T^4 / pi                  : {SB_expected:.4f} W/(m2 sr)")
print(f"  Relative error                  : {abs(L_total/SB_expected-1)*100:.3f}%")

# -- 3. In-band radiance --------------------------------------------------------
print()
print("=== In-band radiance [W/(m2 sr)] ===")
print(f"  {'Band [um]':>10}  {'T [K]':>7}  {'eps':>5}  {'L':>12}")
print("  " + "-" * 42)
band_cases = [
    (8.0, 12.0, 300, 1.0), (8.0, 12.0, 350, 1.0),
    (8.0, 12.0, 250, 1.0), (8.0, 14.0, 300, 1.0),
    (8.0, 12.0, 300, 0.9),
]
for l1, l2, T, eps in band_cases:
    L = in_band_radiance(T, eps, l1, l2)
    print(f"  {l1:.0f}-{l2:.0f} um  {T:>7.0f}  {eps:>5.2f}  {L:>12.4f}")

# -- 4. Solid angle -------------------------------------------------------------
print()
print("=== Solid angle [sr] for R=1.5m ===")
print(f"  {'Range [km]':>12}  {'Omega [sr]':>14}")
print("  " + "-" * 30)
for d_km in [100, 200, 500, 1000, 2000]:
    Omega = solid_angle_sphere(1.5, d_km * 1e3)
    print(f"  {d_km:>12}  {Omega:>14.4e}")

# -- 5. LWIRSensor basic detection ---------------------------------------------
print()
print("=== LWIRSensor detection (R=1.5 m, 8-12 um band) ===")
sensor = LWIRSensor(
    lambda1_um=8.0, lambda2_um=12.0,
    irradiance_cutoff_W_m2=1e-11,
    atm_absorption=0.85,
    T_sunlit_K=350.0,
    T_eclipse_K=240.0,
    emissivity=0.90,
)
print(sensor.summary())
print()

R_m = 1.5   # 1.5 m effective radius
print(f"  {'Range [km]':>12}  {'E sunlit [W/m2]':>16}  "
      f"{'Det?':>5}  {'E eclipse [W/m2]':>17}  {'Det?':>5}")
print("  " + "-" * 63)
for d_km in [100, 200, 500, 1000, 2000, 5000]:
    rs = sensor.detect(R_m, d_km*1e3, in_eclipse=False)
    re = sensor.detect(R_m, d_km*1e3, in_eclipse=True)
    print(f"  {d_km:>12}  {rs.irradiance_W_m2:>16.3e}  "
          f"{'YES' if rs.detected else 'no':>5}  "
          f"{re.irradiance_W_m2:>17.3e}  "
          f"{'YES' if re.detected else 'no':>5}")

# -- 6. Max detection range -----------------------------------------------------
print()
print("=== Maximum detection range ===")
print(f"  {'R [m]':>8}  {'Sunlit [km]':>12}  {'Eclipse [km]':>13}")
print("  " + "-" * 38)
for R in [0.5, 1.0, 1.5, 3.0, 5.0]:
    d_sun = sensor.max_range(R, in_eclipse=False) / 1e3
    d_ecl = sensor.max_range(R, in_eclipse=True)  / 1e3
    print(f"  {R:>8.1f}  {d_sun:>12.1f}  {d_ecl:>13.1f}")

# -- 7. Margin at 500 km, R=1.5 m ----------------------------------------------
print()
result = sensor.detect(R_m=1.5, d_m=500e3, in_eclipse=False)
print(f"=== Detection at 500 km, sunlit, R=1.5 m ===")
print(f"  L (in-band radiance) : {result.in_band_radiance_W_m2_sr:.4f} W/(m2 sr)")
print(f"  Solid angle          : {result.solid_angle_sr:.4e} sr")
print(f"  Irradiance           : {result.irradiance_W_m2:.4e} W/m2")
print(f"  Cutoff               : {sensor.irradiance_cutoff_W_m2:.4e} W/m2")
print(f"  Margin               : {result.margin_dB:.1f} dB")
print(f"  Detected             : {result.detected}")

# -- 8. Emissivity sensitivity --------------------------------------------------
print()
print("=== Emissivity sensitivity (T=350K, 500 km, R=1.5 m) ===")
print(f"  {'epsilon':>8}  {'L [W/(m2 sr)]':>16}  {'E [W/m2]':>14}  {'Margin [dB]':>12}")
print("  " + "-" * 56)
for eps in [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]:
    r = sensor.detect(R_m=1.5, d_m=500e3, in_eclipse=False, emissivity=eps)
    print(f"  {eps:>8.2f}  {r.in_band_radiance_W_m2_sr:>16.4f}  "
          f"{r.irradiance_W_m2:>14.4e}  {r.margin_dB:>12.1f}")

print()
print("All tests complete.")
