"""Verify the exact spectral density by integrating the original scalar."""

import cmath
import math
from pathlib import Path

from check_connection import evolve


def exterior_spectrum(d, w, step=0.002, horizon=16, log_boundary=-28):
    """Ingoing zero-momentum response at w=omega/2, divided by the vacuum."""
    a = (d - 2) / 2
    xi = a * (a - 2) / ((d - 3) * (d - 4))
    bare_mass_sq = -a * (a + 2) + xi * d * (d + 1)

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
    source = field / math.tanh(math.exp(log_boundary)) ** (a / 2)
    return flux / (math.pi * w**2 * abs(source)**2), flux


def exact_spectrum(w):
    q = math.exp(-math.pi * w)
    return (1 - q*q) / (1 - 2*q*math.cos(math.pi*w) + q*q)


def bounce(w, n):
    return 2 * math.exp(-n * math.pi * w) * math.cos(n * math.pi * w)


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
    Path(__file__).with_name("lovelock_rows.tex").write_text(
        "\n".join(rows + [r"\hline"]) + "\n", encoding="ascii"
    )


if __name__ == "__main__":
    main()
