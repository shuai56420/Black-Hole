"""Compute the scalar core connection in the Bessel scaling variable.

Use Python 3. The code integrates on real rays after rotating the
coupling, and uses no external packages.
"""

import cmath
import math
from pathlib import Path


C_BESSEL = math.log(2) - 0.5772156649015328606


def evolve(rhs, start, stop, y, max_step):
    """Fourth-order Runge--Kutta for a pair of complex components."""
    steps = max(1, math.ceil(abs(stop - start) / max_step))
    h = (stop - start) / steps
    for j in range(steps):
        t = start + j * h
        k1 = rhs(t, *y)
        k2 = rhs(t + h / 2, *(a + h * b / 2 for a, b in zip(y, k1)))
        k3 = rhs(t + h / 2, *(a + h * b / 2 for a, b in zip(y, k2)))
        k4 = rhs(t + h, *(a + h * b for a, b in zip(y, k3)))
        y = tuple(
            a + h / 6 * (b + 2 * c + 2 * d + e)
            for a, b, c, d, e in zip(y, k1, k2, k3, k4)
        )
    return y


def tail_action(p, g, radius):
    """Integral of sqrt(Q)-1+g*t**(p-2)/2 from radius to infinity."""
    s = p - 2
    tail = -1 / (8 * radius)
    coefficient = 0.5
    for n in range(2, 15):
        coefficient *= (1.5 - n) / n
        for k in range(n + 1):
            power = k * s - 2 * (n - k)
            tail += (
                coefficient * math.comb(n, k) * (-g)**k * (-0.25)**(n-k)
                * radius ** (power + 1) / (-power - 1)
            )
    return tail


def decaying_ray(p, g, radius=240, step=0.025):
    """Return b and c in F=b*(-log(t)+c) for a normalized decaying mode."""
    s = p - 2
    q = 1 - g * radius**s - 0.25 / radius**2
    dq = -g * s * radius ** (s - 1) + 0.5 / radius**3

    # Evolve F divided by its leading asymptotic. Include the convergent
    # tail integral and the leading WKB amplitude in the data at radius.
    amp = q ** (-0.25) * cmath.exp(tail_action(p, g, radius))
    velocity = (
        1 - 0.5 * g * radius**s - cmath.sqrt(q) - dq / (4 * q)
    ) * amp

    def scaled_equation(t, y, dy):
        potential = (
            0.25 / t**2 + 0.5 * s * g * t ** (s - 1)
            + 0.25 * g * g * t ** (2 * s)
        )
        return dy, (2 - g * t**s) * dy - potential * y

    value, derivative = evolve(scaled_equation, radius, 1, (amp, velocity), step)
    leading = math.sqrt(math.pi / 2) * cmath.exp(-1 + g / (2 * (p - 1)))
    value, derivative = leading * value, leading * (derivative + (-1.5 + g / 2) * value)

    # In log(t) the two local solutions tend to a constant and log(t).
    def logarithmic_equation(u, y, dy):
        return dy, (math.exp(2 * u) - g * math.exp(p * u)) * y

    log_end = -29.0
    value, derivative = evolve(
        logarithmic_equation, 0, log_end, (value, derivative), step
    )
    b = -derivative
    return b, value / b + log_end


def ray_data(d, g, k, radius=240, step=0.025):
    p = d / (d - 1)
    b, c = decaying_ray(p, g * cmath.exp(1j * math.pi * p * k), radius, step)
    return b, c + 1j * math.pi * k


def connection(d, g, radius=240, step=0.025):
    (b_minus, c_minus), (b_zero, c_zero), (b_plus, c_plus) = (
        ray_data(d, g, k, radius, step) for k in (-1, 0, 1)
    )
    target = 1j * math.pi
    adjacent = (
        b_minus * b_zero * (c_zero - c_minus),
        b_zero * b_plus * (c_plus - c_zero),
    )
    ratio = (b_minus + b_plus) / b_zero
    wronskian = b_minus * b_plus * (c_plus - c_minus) / target
    residual = max(abs(x / target - 1) for x in adjacent)
    return ratio, wronskian, residual


def coefficients(d):
    p = d / (d - 1)
    moment = 2**(p - 3) * math.gamma(p / 2)**4 / math.gamma(p)
    b_prime = -moment * math.tan(math.pi * p / 2) / math.pi
    s_prime = 2 * b_prime * (math.cos(math.pi * p) - 1)
    return moment, b_prime, s_prime


def matrix_product(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(2)) for j in range(2))
        for i in range(2)
    )


def compare_full_winding(d, g):
    turns = 2 * (d - 1)
    # The ray coupling repeats after a full turn in the radial coordinate.
    data = [ray_data(d, g, k, radius=240, step=0.018) for k in range(turns)]
    product = ((1, 0), (0, 1))
    for k in range(turns):
        multiplier = (data[k-1][0] + data[(k+1) % turns][0]) / data[k][0]
        product = matrix_product(((multiplier, -1), (1, 0)), product)

    # Compare acting on the two independent Frobenius coefficients, not
    # just the trace. F_{k+turns} shifts c_k by i*pi*turns.
    b0, c0 = data[0]
    bm, cm = data[-1]
    cm -= 1j * math.pi * turns
    frobenius = ((b0 * c0, -b0), (bm * cm, -bm))
    transported = matrix_product(product, frobenius)
    shifted = (
        (b0 * (c0 + 1j * math.pi * turns), -b0),
        (bm * (cm + 1j * math.pi * turns), -bm),
    )
    return max(abs(transported[i][j] - shifted[i][j]) for i in range(2) for j in range(2))


def zfmt(value):
    return f"{value.real:.5f}{value.imag:+.5f}i"


def main():
    rows = []
    for d in (4, 5, 6):
        p = d / (d - 1)
        moment, b_prime, s_prime = coefficients(d)
        b, c = decaying_ray(p, 0)
        assert abs(b - 1) < 9e-7 and abs(c - C_BESSEL) < 9e-7
        positive = decaying_ray(p, 1e-3)
        negative = decaying_ray(p, -1e-3)
        db, dc = ((a - b) / 0.002 for a, b in zip(positive, negative))
        ds = (connection(d, 1e-3)[0] - connection(d, -1e-3)[0]) / 0.002
        assert abs(db - b_prime) < 7e-5
        assert abs(dc - moment) < 4e-6
        assert abs(ds - s_prime) < 1.1e-4
        print(f"d={d}: S'(0) numeric={ds.real:.8f}, exact={s_prime:.8f}")

        for mass_parameter in (0.05, 0.30):
            g = mass_parameter * cmath.exp(-0.5j * math.pi * p)
            s1, _, delta1 = connection(d, g, radius=160, step=0.025)
            s2, s_wronskian, delta2 = connection(d, g, radius=320, step=0.014)
            assert abs(s1 - s2) < 3e-5
            assert max(delta1, delta2) < 2e-5
            assert abs(s2 - s_wronskian) < 3e-5
            linear = -(2 + s_prime * g)
            print(f"d={d}, mu={mass_parameter:.2f}: R={-s2:.8g}; "
                  f"adjacent Wronskian residual={delta2:.2e}")
            values = " & ".join("$" + zfmt(x) + "$" for x in (-s2, linear))
            rows.append(f"{d} & {mass_parameter:.2f} & {values} " + r"\\")

        winding_residual = compare_full_winding(d, 0.25)
        assert winding_residual < 3e-4
        print(f"d={d}: full radial winding residual={winding_residual:.2e}")

    Path(__file__).with_name("connection_rows.tex").write_text(
        "\n".join(rows + [r"\hline"]) + "\n", encoding="ascii"
    )


if __name__ == "__main__":
    main()
