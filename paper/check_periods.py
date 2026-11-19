"""Quadrature checks for the radial periods and fixed-time action differences.

Run with Python 3. No third-party modules are required.
"""

import cmath
import math
from pathlib import Path


def integrate(function, tol=3e-12):
    """Integrate a smooth complex function on [0, 1] by adaptive Simpson."""

    def interval(left, right, fa, fm, fb, whole, tolerance, depth):
        midpoint = (left + right) / 2
        fl = function((left + midpoint) / 2)
        fr = function((midpoint + right) / 2)
        sl = (midpoint - left) * (fa + 4 * fl + fm) / 6
        sr = (right - midpoint) * (fm + 4 * fr + fb) / 6
        error = sl + sr - whole
        if abs(error) <= 15 * tolerance:
            return sl + sr + error / 15
        if depth == 0:
            raise ArithmeticError("quadrature did not converge")
        return interval(left, midpoint, fa, fl, fm, sl, tolerance / 2, depth - 1) + (
            interval(midpoint, right, fm, fr, fb, sr, tolerance / 2, depth - 1)
        )

    fa, fm, fb = (function(x) for x in (0, 0.5, 1))
    return interval(0, 1, fa, fm, fb, (fa + 4 * fm + fb) / 6, tol, 24)


def geometric_sum(x, count):
    return sum(x**j for j in range(count))


def turning_point(d, energy, j=0):
    m = d - 2
    scale = energy ** (-2 / m)
    zeta = cmath.exp(2j * math.pi * j / m)
    perturbation = (scale * zeta) ** d
    x = 1 + 0j
    for _ in range(30):
        correction = (x**m + perturbation * x**d - 1) / (
            m * x ** (m - 1) + d * perturbation * x ** (d - 1)
        )
        x -= correction
        if abs(correction) < 3e-15:
            assert abs(x**m + perturbation * x**d - 1) < 2e-13
            return scale * zeta * x, cmath.log(scale * x) + 2j * math.pi * j / m
    raise ArithmeticError("turning-point solver did not converge")


def radial_cap(d, energy, j, quantity):
    """Integrate from the origin to the designated root on the chosen sheet."""
    m = d - 2
    rj, log_rj = turning_point(d, energy, j)
    u = energy**2 * rj**m
    v = rj**d
    prefactor = -2j * cmath.exp((m / 2 + 1) * log_rj)
    if quantity == "time":
        # Extract the small power before quadrature to control relative error.
        prefactor *= energy * rj**m

    def integrand(s):
        x = 1 - s**2
        base = x ** (m / 2) / cmath.sqrt(
            u * geometric_sum(x, m) + v * geometric_sum(x, d)
        )
        if quantity == "time":
            return base * x**m / ((rj * x) ** d - 1)
        return base

    return prefactor * integrate(integrand)


def local_cycle(d, energy):
    return tuple(
        2 * (radial_cap(d, energy, 1, kind) - radial_cap(d, energy, 0, kind))
        for kind in ("length", "time")
    )


def predicted_constants(d):
    m = d - 2
    alpha = d / (2 * m)
    phase = cmath.exp(2j * math.pi * alpha)
    beta = math.gamma(alpha) * math.gamma(0.5) / math.gamma(alpha + 0.5)
    length = -2j * (phase - 1) * beta / m
    ratio = -d / (2 * (d - 1))
    return length, ratio * length, (1 + ratio) * length


def reference_path(d, energy, tol=3e-12):
    """Length and time on the original bounce, with an upper horizon detour."""
    m = d - 2
    r0, _ = turning_point(d, energy)
    r0m = r0 ** (-m)

    def integrand(s, kind):
        if s == 0:
            cap = 2 / cmath.sqrt(2 * r0 + m * r0m / r0)
            return cap if kind == "length" else -cap / energy
        if s == 1:
            return 0 if kind == "length" else -2 / d
        offset = s**2 / (1 - s)
        jacobian = s * (2 - s) / (1 - s) ** 2
        r = r0 + offset
        q = r0 / r
        radial = cmath.sqrt(offset * (r + r0 + r0m * geometric_sum(q, m) / r))
        if kind == "length":
            difference = r0m * q**m - energy**2
            return jacobian * difference / (r * radial * (r + radial))

        # Remove the simple pole via 2/[d(r^2-1)] before integration.
        numerator = sum(r**k * geometric_sum(r, m - k) for k in range(m)) - r**m
        regular_tortoise = numerator / (d * geometric_sum(r, d)) + 1 / (d * (r + 1))
        return jacobian * (regular_tortoise - 1 / (radial * (energy + radial)))

    length = 2 * (integrate(lambda s: integrand(s, "length"), tol) - cmath.log(r0))
    time = 2 * (
        integrate(lambda s: integrand(s, "time"), tol)
        - cmath.log((1 - r0) / (1 + r0)) / d
        - 1j * math.pi / d
    )
    return length, time


def fixed_time_check(d, energy, winding):
    base_length, target = reference_path(d, energy)
    shifted_energy = complex(energy)
    for _ in range(10):
        length, time = reference_path(d, shifted_energy)
        cycle_length, cycle_time = local_cycle(d, shifted_energy)
        residual = time + winding * cycle_time - target
        if abs(residual) < 6e-12:
            return length + winding * cycle_length - base_length, abs(residual)
        h = 2e-4 * energy
        derivative = (
            reference_path(d, shifted_energy + h)[1]
            - reference_path(d, shifted_energy - h)[1]
        ) / (2 * h)
        shifted_energy -= residual / derivative
    raise ArithmeticError("fixed-time energy solver did not converge")


def complex_tex(z):
    return f"{z.real:.6f}{z.imag:+.6f}i"


def main():
    rows = []
    for d in (4, 5, 6, 8):
        power = d / (d - 2)
        predicted_length, predicted_time, predicted_action = predicted_constants(d)
        for energy in (4, 8, 16, 32):
            length, time = local_cycle(d, energy)
            scaled_length, scaled_time = length * energy**power, time * energy ** (power + 1)
            if d == 4:
                assert max(abs(scaled_length), abs(scaled_time)) < 2e-13
                base_length, base_time = reference_path(d, energy)
                exact_length = math.log(4 / math.sqrt(energy**4 + 4))
                exact_time = (math.pi / 2) * (1 - 1j) - integrate(
                    lambda x: 2 * energy**3 / (energy**4 + 4 * x**4)
                )
                assert abs(base_length - exact_length) < 2e-11
                assert abs(base_time - exact_time) < 2e-11
            if energy == 32:
                errors = (abs(scaled_length - predicted_length), abs(scaled_time - predicted_time))
                assert max(errors) < 0.002
            if d in (5, 6) and energy in (8, 32):
                delta, residual = fixed_time_check(d, energy, 1)
                scaled_delta = delta * energy**power
                target_time = 2 * math.pi / d * (1 / math.tan(math.pi / d) - 1j)
                assert abs(energy * (reference_path(d, energy)[1] - target_time) + 2) < 0.2
                assert abs(scaled_delta - predicted_action) < 0.028
                print(f"d={d}, E={energy}: fixed-time residual={residual:.2e}; "
                      f"scaled action={scaled_delta:.9g}, prediction={predicted_action:.9g}")
                rows.append(f"{d} & {energy} & ${complex_tex(scaled_length)}$ & "
                            f"${complex_tex(scaled_time)}$ & ${complex_tex(scaled_delta)}$ \\\\")

        if d in (5, 6):
            rows.append(f"{d} & $\u005cinfty$ & ${complex_tex(predicted_length)}$ & "
                        f"${complex_tex(predicted_time)}$ & ${complex_tex(predicted_action)}$ \\\\")
        if d == 6:
            double_delta, residual = fixed_time_check(d, 16, 2)
            assert abs(double_delta * 16**power - 2 * predicted_action) < 0.03
            print(f"d=6, E=16, two cycles: fixed-time residual={residual:.2e}")

    Path(__file__).with_name("period_rows.tex").write_text(
        "\n".join(rows + [r"\hline"]) + "\n", encoding="ascii"
    )
    print("Local-period, zero-period, and fixed-time checks passed.")


if __name__ == "__main__":
    main()
