"""Compare the finite scaled-momentum Airy connection and Lovelock response."""

import cmath
import math
from decimal import Decimal, localcontext
from pathlib import Path

from check_connection import evolve
from check_lovelock_momentum import numerical_pole
from check_lovelock_momentum_time import OPTICAL_DISTANCE
from check_lovelock_residues import numerical_response


ROTATION = cmath.exp(2j*math.pi/3)


def airy_ray(coupling, cutoff, step, terms, shift=0j):
    """Return the origin data of the mode normalized to Ai(X - shift)."""
    coefficients = [1+0j]
    for j in range(terms-1):
        correction = (2*j+1)*(2*j-3)*coefficients[j-2]/16 if j >= 2 else 0j
        momentum = sum((-shift)**power*coefficients[j-2*power] for power in range(j//2+1))
        coefficients.append(-(coupling*momentum+correction)/(j+1))

    v = (cutoff-shift)**(-0.5)
    value = sum(coefficient*v**j for j, coefficient in enumerate(coefficients))
    derivative = sum(-0.5*j*coefficient*v**(j+2) for j, coefficient in enumerate(coefficients))

    def scaled_equation(x, y, dy):
        shifted = x-shift
        return dy, ((2*cmath.sqrt(shifted)+0.5/shifted)*dy
                    -(coupling/x+5/(16*shifted**2))*y)

    value, derivative = evolve(scaled_equation, cutoff, 1, (value, derivative), step)
    shifted = 1-shift
    prefactor = shifted**(-0.25)*cmath.exp(-2*shifted**1.5/3)/(2*math.sqrt(math.pi))
    derivative = prefactor*(derivative-(cmath.sqrt(shifted)+0.25/shifted)*value)
    value *= prefactor

    def end_equation(z, y, dy):
        x = math.exp(z)
        return x*dy, (x*(x-shift)-coupling)*y

    log_end = -29
    value, derivative = evolve(end_equation, 0, log_end, (value, derivative), step)
    return value, derivative+coupling*value*log_end


def connection_multiplier(coupling, cutoff=18, step=0.015, terms=18, shift=0j):
    """Normalized Stokes multiplier and adjacent Wronskian residual."""
    def ray(index):
        value, regular_derivative = airy_ray(
            coupling*ROTATION**index, cutoff, step, terms, shift*ROTATION**(-index)
        )
        derivative = ROTATION**(-index)*regular_derivative
        derivative += coupling*value*index*2j*math.pi/3
        return value, derivative

    (left, d_left), (middle, d_middle), (right, d_right) = (ray(j) for j in (-1, 0, 1))
    adjacent = -math.sqrt(3)/(6*math.pi)*(ROTATION**(-1)-1)
    residual = max(
        abs((middle*d_right-d_middle*right)/adjacent-1),
        abs((left*d_middle-d_left*middle)/(ROTATION*adjacent)-1),
    )
    return -(ROTATION*(left+ROTATION*right))/middle, residual


def matched_harmonics(nu, pole, momentum_squared, radius, samples, point, terms):
    dimension = float(nu)
    phase = 1-cmath.exp(2j*math.pi*dimension)
    harmonics = [0j, 0j]
    for sample in range(samples):
        theta = math.tau*(sample+0.5)/samples-math.pi
        symbol = radius*cmath.exp(1j*theta)
        omega = pole+(theta-1j*math.log(radius))/(2*OPTICAL_DISTANCE)
        vacuum = (math.gamma(-dimension)/math.gamma(dimension)
                  * cmath.exp(-1j*math.pi*dimension)*(omega/2)**(2*dimension)
                  * (1-float(momentum_squared)/omega**2)**dimension)
        response = numerical_response(omega, nu, momentum_squared, point, terms)
        normalized = (response/vacuum-1)/phase
        for slot, harmonic in enumerate((2, 3)):
            harmonics[slot] += normalized/(samples*symbol**harmonic)
    return harmonics


def main():
    zero, zero_error = connection_multiplier(0)
    assert max(zero_error, abs(zero-1)) < 3e-8
    step = 0.001
    plus, _ = connection_multiplier(step)
    minus, _ = connection_multiplier(-step)
    linear = math.gamma(1/3)**2/3**(1/3)
    assert abs((plus-minus)/(2*step)-linear) < 0.00006

    rows = []
    for scaled_momentum in (0.2, 0.45):
        coupling = cmath.exp(-1j*math.pi/3)*scaled_momentum/(1-1j)**(2/3)
        amplitude, residual = connection_multiplier(coupling)
        alternate, alternate_residual = connection_multiplier(coupling, 27, 0.008, 24)
        assert max(residual, alternate_residual, abs(amplitude-alternate)) < 1e-7
        radius = 0.5/abs(amplitude)
        predicted_shift = 1j*cmath.log(amplitude)/(2*OPTICAL_DISTANCE)
        print(f"scaled momentum={scaled_momentum}: Airy={amplitude}, shift={predicted_shift}, "
              f"Wronskian error={residual:.2e}", flush=True)
        for nu in (Decimal('0.75'), Decimal('1.25')):
            errors = [math.inf]*3
            with localcontext() as context:
                context.prec = 240
                for n in (12, 32, 64):
                    N = n+(1+nu)/2
                    pole = complex(2*N, -2*N)
                    momentum_squared = 4*Decimal(str(scaled_momentum))*N**(Decimal(2)/3)
                    direct_pole = pole
                    point, terms = Decimal('0.56'), 820 if n == 64 else 520
                    for part in range(1, 5):
                        direct_pole = numerical_pole(
                            nu, momentum_squared*part/4, direct_pole, point, terms
                        )
                    matched = matched_harmonics(nu, pole, momentum_squared, radius, 32, point, terms)
                    if n == 64:
                        refined_pole = numerical_pole(
                            nu, momentum_squared, direct_pole, Decimal('0.53'), 900
                        )
                        refined_harmonics = matched_harmonics(
                            nu, pole, momentum_squared, radius, 48, Decimal('0.53'), 900
                        )
                        assert abs(direct_pole-refined_pole) < 1e-8
                        assert max(abs(a/b-1) for a, b in zip(matched, refined_harmonics)) < 0.001
                    refined = [abs(direct_pole-pole-predicted_shift)]
                    refined += [abs(value/amplitude**harmonic-1)
                                for value, harmonic in zip(matched, (2, 3))]
                    print(f"nu={nu}, n={n}, k^2/N^(2/3)={4*scaled_momentum}: "
                          f"errors={refined}", flush=True)
                    assert all(new < old for new, old in zip(refined, errors))
                    errors = refined
                    cells = " & ".join(f"{error:.3e}" for error in errors)
                    rows.append(f"{scaled_momentum:g} & {nu:g} & {n} & {cells} " + r"\\")
            assert errors[0] < 0.002 and max(errors[1:]) < 0.01
    Path(__file__).with_name("lovelock_scaled_momentum_rows.tex").write_text(
        "\n".join(rows + [r"\hline"]) + "\n", encoding="ascii"
    )


if __name__ == "__main__":
    main()
