"""Check momentum harmonics, high-overtone residues, and the phase sum."""

import cmath
import math
from decimal import Decimal, localcontext
from pathlib import Path

from check_lovelock_momentum import analytic_slope
from check_lovelock_residues import (
    laurent_pair,
    numerical_response_slope,
    zero_momentum_residue,
)


OPTICAL_DISTANCE = math.pi*(1+1j)/4


def airy_log_derivative(omega):
    return (
        cmath.exp(-1j*math.pi/3)*math.gamma(1/3)**2
        / (4*3**(1/3)*(omega/2)**(2/3))
    )


def response_harmonics(nu, pole, samples):
    radius = 0.45
    order = float(nu)
    normalization = 2j*math.pi/(math.gamma(order)*math.gamma(1+order))
    values = [0j, 0j]
    for sample in range(samples):
        theta = math.tau*sample/samples
        symbol = radius*cmath.exp(1j*theta)
        omega = pole+(theta-1j*math.log(radius))/(2*OPTICAL_DISTANCE)
        slope = numerical_response_slope(omega, nu, terms=420)
        normalized = slope/(normalization*(omega/2)**(2*order)*airy_log_derivative(omega))
        for slot, harmonic in enumerate((2, 3)):
            values[slot] += normalized/(samples*harmonic*symbol**harmonic)
    return values


def check_harmonics():
    rows = []
    with localcontext() as context:
        context.prec = 145
        for nu in (Decimal('0.75'), Decimal('1.25')):
            previous = (math.inf,)*3
            for n in (8, 16, 32):
                pole, shift = analytic_slope(n, nu)
                residue = zero_momentum_residue(float(nu), n)
                double, simple = laurent_pair(
                    lambda omega: numerical_response_slope(omega, nu, terms=420),
                    pole, 0.35, 12
                )
                assert abs(double/(residue*shift)-1) < 1e-7
                errors = [abs(value-1) for value in response_harmonics(nu, pole, 20)]
                errors.append(abs(simple/(residue*shift)))
                assert all(new < old for new, old in zip(errors, previous))
                previous = errors
                values = " & ".join(f"{error:.3e}" for error in errors)
                rows.append(f"{nu:g} & {n} & {values} " + r"\\")
                print(f"nu={nu}, n={n}: harmonic errors and residue ratio {errors}", flush=True)
            assert max(previous[:2]) < 0.003 and previous[2] < 0.02
    return rows


def check_displacement_sum():
    """Test the primitive coefficient using the exact pole-motion terms."""
    nu = Decimal('0.75')
    dimension = float(nu)
    h = (1+dimension)/2
    power = 2*dimension+1/3
    residue_asymptotic = (
        -4*2**dimension*cmath.exp(-0.5j*math.pi*dimension)
        / ((1+1j)*math.gamma(dimension)*math.gamma(1+dimension))
    )
    slope_asymptotic = 1j*airy_log_derivative(2*(1-1j))/(2*OPTICAL_DISTANCE)
    modes = []
    with localcontext() as context:
        context.prec = 140
        for n in range(320):
            _, slope = analytic_slope(n, nu)
            modes.append(zero_momentum_residue(dimension, n)*slope)
    rows = []
    for harmonic in (2, 3):
        image_time = harmonic*math.pi/2*(1+1j)
        leading = (
            -image_time*residue_asymptotic*slope_asymptotic*math.gamma(power)
            * cmath.exp(-2j*math.pi*harmonic*h)
        )
        errors = []
        for epsilon in (0.15, 0.075, 0.035):
            time = image_time+epsilon
            phase = -time*sum(
                value*cmath.exp(-2*(1+1j)*(n+h)*time)
                + value.conjugate()*cmath.exp(-2*(1-1j)*(n+h)*time)
                for n, value in enumerate(modes)
            )
            scaled = phase*(2*(1+1j)*epsilon)**power
            errors.append(abs(scaled/leading-1))
        assert errors[2] < errors[1] < errors[0] and errors[2] < 0.035
        values = " & ".join(f"{error:.3e}" for error in errors)
        rows.append(f"{harmonic} & {values} " + r"\\")
        print(f"m={harmonic}: phase sum coefficient errors {errors}", flush=True)
    return rows


def main():
    for filename, rows in (
        ("lovelock_momentum_harmonic_rows.tex", check_harmonics()),
        ("lovelock_momentum_time_rows.tex", check_displacement_sum()),
    ):
        Path(__file__).with_name(filename).write_text(
            "\n".join(rows + [r"\hline"]) + "\n", encoding="ascii"
        )


if __name__ == "__main__":
    main()
