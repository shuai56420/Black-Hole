"""Verify the exact spectral density by integrating the original scalar."""

import cmath
import math
from pathlib import Path

from check_connection import evolve


def exterior_spectrum(d, w, step=0.002, horizon=16, log_boundary=-28, nu=1):
    """Ingoing zero-momentum response at w=omega/2, divided by the vacuum."""
    a = (d - 2) / 2
    xi = a * (a - 2) / ((d - 3) * (d - 4))
    bare_mass_sq = nu**2 - 1 - a * (a + 2) + xi * d * (d + 1)

    def coefficients(z):
        inverse_r_sq = math.tanh(z)
        f = 2 / math.sinh(2 * z)
        ricci = -d * (d + 1) + (d - 3) * (d - 4) * inverse_r_sq**2
        effective_mass_sq = bare_mass_sq + xi * ricci
        return a * f, (w**2 - f * effective_mass_sq / 4) / inverse_r_sq

    # Use the original nonminimally coupled field, without the analytic
    # reduction to its hypergeometric radial amplitude.
    def finite_z(z, phi, derivative):
        friction, frequency = coefficients(z)
        return derivative, friction * derivative - frequency * phi

    ingoing = cmath.exp(1j * w * horizon)
    field, derivative = evolve(
        finite_z, horizon, 1, (ingoing, 1j * w * ingoing), step
    )
    flux = (field.conjugate() * derivative).imag / math.tanh(1) ** a

    # At the boundary z approaches zero. The log coordinate keeps the
    # coefficients bounded and resolves the incoming solution's falloff.
    def log_z(s, phi, derivative):
        z = math.exp(s)
        friction, frequency = coefficients(z)
        return derivative, (1 + z * friction) * derivative - z**2 * frequency * phi

    field, _ = evolve(log_z, 0, log_boundary, (field, derivative), step)
    source = field / math.tanh(math.exp(log_boundary)) ** ((a + 1 - nu) / 2)
    vacuum_flux = math.pi * w ** (2 * nu) / math.gamma(nu)**2
    return flux / (vacuum_flux * abs(source)**2), flux


def log_gamma(z):
    """Stirling expansion after recurrence into the right half-plane."""
    shift = 0j
    while z.real < 17:
        shift += cmath.log(z)
        z += 1
    inverse = 1 / z
    return (
        (z - 0.5) * cmath.log(z) - z + 0.5 * math.log(2 * math.pi)
        + inverse / 12 - inverse**3 / 360 + inverse**5 / 1260
        - inverse**7 / 1680 + inverse**9 / 1188 - shift
    )


def spectral_envelope(w, nu):
    h = (1 + nu) / 2
    s = (1 + 1j) * w / 2
    gamma_ratio = 2 * (log_gamma(h + s) - log_gamma(1 - h + s)).real
    return math.exp(gamma_ratio - 2 * nu * math.log(abs(s)))


def reflection_angle(w, nu):
    return math.pi * (w - nu - 1)


def exact_spectrum(w, nu=1):
    q = math.exp(-math.pi * w)
    phase = reflection_angle(w, nu)
    return spectral_envelope(w, nu) * (1 - q*q) / (1 - 2*q*math.cos(phase) + q*q)


def bounce(w, n, nu=1):
    return 2 * math.exp(-n * math.pi * w) * math.cos(n * reflection_angle(w, nu))


def table(name, rows):
    Path(__file__).with_name(name).write_text(
        "\n".join(rows + [r"\hline"]) + "\n", encoding="ascii"
    )


def main():
    rows = []
    max_error = 0
    for d in (8, 12):
        for w in (0.65, 1.0, 1.4, 1.8):
            ratio, flux = exterior_spectrum(d, w)
            refined, refined_flux = exterior_spectrum(d, w, 0.001, 17, -30)
            error = abs(refined - exact_spectrum(w))
            max_error = max(max_error, error)
            assert abs(refined_flux - w) < 3e-9 and abs(flux - w) < 3e-8
            assert abs(refined - ratio) < 2e-8 and error < 2e-9
            after_one = refined - 1 - bounce(w, 1)
            second = bounce(w, 2)
            tail_bound = 2 * math.exp(-3 * math.pi * w) / (1 - math.exp(-math.pi * w))
            assert abs(after_one - second) < tail_bound + 2e-9
            print(f"D={d+1:2d}, omega={2*w:.2f}: exact residual={error:.2e}; "
                  f"after first term={after_one:+.8e}; two-bounce term={second:+.8e}")
            if d == 8 and w >= 1:
                values = " & ".join(f"{x:+.8f}" for x in (after_one, second, tail_bound))
                rows.append(f"{2*w:.1f} & {values} " + r"\\")
    print(f"Maximum ingoing-integration error={max_error:.2e}")
    table("lovelock_rows.tex", rows)

    detuned = []
    detuned_error = 0
    for nu in (0.75, 1.5, 2, 3):
        for w in (1.1, 1.5, 1.8):
            if nu == 2:
                assert abs(spectral_envelope(w, nu) - (1 + 0.25 / w**4)) < 2e-13
            if nu == 3:
                assert abs(spectral_envelope(w, nu) - (1 + 4 / w**4)) < 2e-13
            coarse, _ = exterior_spectrum(8, w, log_boundary=-35, nu=nu)
            ratio, flux = exterior_spectrum(8, w, 0.001, 17, -37, nu)
            exact = exact_spectrum(w, nu)
            error = abs(exact - ratio)
            detuned_error = max(detuned_error, error)
            assert abs(flux - w) < 3e-9
            assert abs(coarse - ratio) < 2e-8 and error < 5e-9
            envelope = spectral_envelope(w, nu)
            after_one = ratio / envelope - 1 - bounce(w, 1, nu)
            bound = 2 * math.exp(-3 * math.pi * w) / (1 - math.exp(-math.pi * w))
            second = bounce(w, 2, nu)
            assert abs(after_one - second) < bound + 5e-9
            if w == 1.1:
                values = " & ".join(f"{x:+.8f}" for x in (ratio, after_one, second))
                detuned.append(f"{nu:g} & {values} " + r"\\")
            print(f"D=9, nu={nu:g}, omega={2*w:.2f}: residual={error:.2e}; "
                  f"first harmonic={bounce(w, 1, nu):+.7e}; second={second:+.7e}")
    print(f"Maximum detuned integration error={detuned_error:.2e}")
    table("lovelock_detuned_rows.tex", detuned)


if __name__ == "__main__":
    main()
