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


def outer_turning_point(d, energy):
    """Root that approaches +iE, obtained in the exterior scale r/E."""
    m = d - 2
    perturbation = energy**(-d)
    x = 1j
    for _ in range(12):
        step = (1+x*x-perturbation*x**(-m)) / (2*x+m*perturbation*x**(-m-1))
        x -= step
        if abs(step) < 1e-15:
            return energy*x
    raise ArithmeticError("outer turning point did not converge")


def outer_tail(d, energy, root):
    """Doubled integrals from the outer root to positive-real infinity."""
    eps = root**(-d)

    def inverse_radius(s, time):
        u = 1-s*s
        regular = cmath.sqrt(1+u+eps*u*u*geometric_sum(u, d-2))
        if time:
            return 2*u / ((1-eps*u**d)*regular)
        return 2*u*(1-eps*(1-u**(d-2))) / (regular*(1+s*regular))

    length = 2*(integrate(lambda s: inverse_radius(s, False)) - cmath.log(root))
    time = 2*energy/root**2 * integrate(lambda s: inverse_radius(s, True))
    return length, time


def optical_cycle_quadrature(d, energy, root, radius, detour_side=1):
    """Integrate the finite cycle along a horizon detour and an exterior arc."""
    m = d - 2
    inner, _ = turning_point(d, energy)
    phi = cmath.phase(root)
    outgoing_span = 0.8-inner.real
    incoming_span = radius/abs(root)-1

    def straight_cap(s):
        r = inner+outgoing_span*s*s
        regular = cmath.sqrt(r+inner+inner**(-m)*geometric_sum(inner/r, m)/r)
        return r, 2*math.sqrt(outgoing_span)/regular

    def horizon_detour(s):
        offset = 0.2*cmath.exp(detour_side*1j*math.pi*(1-s))
        r = 1+offset
        return r, -detour_side*1j*math.pi*offset/cmath.sqrt(energy**2+r*r-r**(-m))

    def real_exterior(s):
        r = 1.2+(radius-1.2)*s
        return r, (radius-1.2)/cmath.sqrt(energy**2+r*r-r**(-m))

    def outer_arc(s):
        r = radius*cmath.exp(1j*phi*s)
        return r, 1j*phi/cmath.sqrt(1+energy**2/r**2-r**(-d))

    def outer_cap(s):
        y = 1+incoming_span*s*s
        regular = cmath.sqrt(y+1+root**(-d)*geometric_sum(1/y, m)/y)
        return root*y, -2*math.sqrt(incoming_span)/regular

    paths = (straight_cap, horizon_detour, real_exterior, outer_arc, outer_cap)

    def time_form(path, s):
        r, length_form = path(s)
        return energy*length_form/(r*r-r**(-m))

    length = 2*sum(integrate(lambda s: path(s)[1]) for path in paths)
    time = 2*sum(integrate(lambda s: time_form(path, s)) for path in paths)
    return length, time


def check_optical_cycles():
    rows = []
    for d, bounces in ((4, 2), (5, 2), (6, 3), (8, 3)):
        m = d-2
        power = d/m
        optical_time = 2*math.pi/d * (1/math.tan(math.pi/d)-1j)
        coefficient = 0 if d == 4 else (
            2/m*math.gamma(-1/m)*math.gamma(0.5)/math.gamma(0.5-1/m)
        )
        target_length = bounces*coefficient
        target_time = -bounces*power/(power+1)*coefficient
        errors = []
        fixed_time_errors = []
        for energy in (8, 32, 128):
            root = outer_turning_point(d, energy)
            ref_length, ref_time = reference_path(d, energy)
            tail_length, tail_time = outer_tail(d, energy, root)
            cycle_length = ref_length-tail_length
            cycle_time = ref_time-tail_time
            quadrature_length, quadrature_time = optical_cycle_quadrature(d, energy, root, 2*energy)
            residual = max(abs(cycle_length-quadrature_length), abs(cycle_time-quadrature_time))
            assert residual < 5e-11
            lower_length, lower_time = optical_cycle_quadrature(d, energy, root, 3*energy, -1)
            assert abs(lower_length-cycle_length) < 5e-11
            assert abs(lower_time-cycle_time-4j*math.pi/d) < 5e-11
            if d == 4:
                assert abs(cycle_length-1j*math.pi) < 2e-11
                assert abs(cycle_time-optical_time) < 2e-11
                print(f"d=4, E={energy}: exact cycle residual="
                      f"{max(abs(cycle_length-1j*math.pi), abs(cycle_time-optical_time)):.2e}")
                continue

            length = ref_length+(bounces-1)*cycle_length
            time = ref_time+(bounces-1)*cycle_time
            scaled_length = energy**power*(length-2*math.log(2/energy)-1j*math.pi*(bounces-1))
            scaled_time = energy**(power+1)*(time-bounces*optical_time+2/energy)
            delta = bounces*optical_time-time
            fixed_time = (length-2*cmath.log(delta)-1j*math.pi*(bounces-1)) / (delta/2)**power
            fixed_time_errors.append(abs(fixed_time-target_length/(power+1)))
            assert max(abs(scaled_length.imag), abs(scaled_time.imag)) < 0.1
            error = max(abs(scaled_length-target_length), abs(scaled_time-target_time))
            errors.append(error)
            if energy <= 32:
                rows.append(f"{d} & {bounces} & {energy} & ${complex_tex(scaled_length)}$ & "
                            f"${complex_tex(scaled_time)}$ " + r"\\")
            print(f"d={d}, bounces={bounces}, E={energy}: "
                  f"scaled length={scaled_length:.8g}, scaled time={scaled_time:.8g}, "
                  f"direct-contour residual={residual:.2e}, asymptotic error={error:.2e}")
        if d > 4:
            assert errors[1] < errors[0] and errors[1] < 0.0002
            assert fixed_time_errors[2] < fixed_time_errors[1] < fixed_time_errors[0]
            assert fixed_time_errors[2] < 0.003
            print(f"d={d}: fixed-time scaled action residual at E=128: {fixed_time_errors[2]:.2e}")
            rows.append(f"{d} & {bounces} & $\\infty$ & ${target_length:.6f}$ & "
                        f"${target_time:.6f}$ " + r"\\")

    Path(__file__).with_name("optical_cycle_rows.tex").write_text(
        "\n".join(rows + [r"\hline"]) + "\n", encoding="ascii"
    )


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
    check_optical_cycles()


if __name__ == "__main__":
    main()
