"""Test the displaced Airy coefficient against the exact Lovelock pole slopes."""

import cmath
import math
from decimal import Decimal, localcontext
from pathlib import Path

from check_einstein_poles import ComplexDecimal
from check_lovelock_momentum import analytic_slope
from check_periods import integrate


def gamma_third(numerator):
    """Gamma(numerator/3) from the exponential series on [0,180]."""
    with localcontext() as context:
        context.prec = 190
        a = Decimal(numerator)/3
        limit = Decimal(180)
        term = Decimal(1)
        value = 1/a
        for j in range(1, 1200):
            term = -limit*term/j
            increment = term/(a+j)
            value += increment
            if abs(increment) < Decimal('1e-100'):
                return limit**a*value
    raise ArithmeticError("Gamma series did not converge")


def airy_initial_data():
    cube_root = Decimal(3)**(Decimal(1)/3)
    return (1/(cube_root**2*gamma_third(2)), -1/(cube_root*gamma_third(1)), Decimal(3).sqrt())


def airy_values(z, initial):
    """Ai and Bi from the two nonzero Maclaurin series of the Airy equation."""
    f0 = term0 = ComplexDecimal(1)
    f1 = term1 = z
    z_cubed = z*z*z
    for j in range(1, 420):
        index = 3*j
        term0 = term0*z_cubed/(index*(index-1))
        term1 = term1*z_cubed/((index+1)*index)
        f0 = f0+term0
        f1 = f1+term1
        if max(abs(complex(term0)), abs(complex(term1))) < 1e-80:
            break
    else:
        raise ArithmeticError("Airy series did not converge")
    ai0, aip0, sqrt3 = initial
    return f0*ai0+f1*aip0, (f0*ai0-f1*aip0)*sqrt3


def airy_origin_derivative(displacement, initial, cutoff, tail_terms):
    """Derivative at g=0 of the Airy solution value at the origin."""
    minus_displacement = ComplexDecimal(-displacement.real, -displacement.imag)
    ai0, bi0 = airy_values(minus_displacement, initial)

    def integrand(s):
        if s == 0:
            return -cutoff*complex(ai0)
        ai, bi = airy_values(minus_displacement + Decimal(cutoff*s), initial)
        kernel = ai*(bi0*ai-ai0*bi)
        return math.pi*complex(kernel)/s

    # The Ai^2 tail is exponentially small. For Ai Bi, use the
    # asymptotic series from P'''-4z P'-2P=0 and t=cutoff/u**2.
    coefficients = [1.0]
    for j in range(1, tail_terms):
        power = -0.5-3*(j-1)
        coefficients.append(-coefficients[-1]*power*(power-1)*(power-2)/(12*j))

    def tail_integrand(u):
        denominator = cutoff-displacement*u*u
        ratio = u*u/denominator
        return sum(value*ratio**(3*j) for j, value in enumerate(coefficients))/cmath.sqrt(denominator)

    magnitude = abs(complex(ai0))
    direct = integrate(integrand, 2e-9*max(1, magnitude))
    tail = complex(ai0)*integrate(tail_integrand, 3e-12)
    return direct-tail, complex(ai0)


def crossover_coefficient(scaled_mass, initial, cutoff=13, tail_terms=6):
    displacement = cmath.exp(-1j*math.pi/3)*scaled_mass**2/(4*2**(2/3))
    rotations = [cmath.exp(2j*math.pi*j/3) for j in range(3)]
    sector_data = [
        airy_origin_derivative(displacement*rotation, initial, cutoff, tail_terms)
        for rotation in rotations
    ]
    log_derivative = -sum(value[0] for value in sector_data)/sector_data[0][1]
    optical = math.pi*(1+1j)/4
    return 1j*cmath.exp(-1j*math.pi/3)*log_derivative/(8*optical*(1-1j)**(2/3))


def slope_at_scaled_mass(n, scaled_mass):
    """Select nu so that sqrt(nu**2-1)/(n+h)**(2/3) is held fixed."""
    mass = Decimal(scaled_mass)
    nu = Decimal(1)
    for _ in range(320):
        N = n+(1+nu)/2
        new_nu = (1+mass*mass*N**(Decimal(4)/3)).sqrt()
        if abs(nu-new_nu) < Decimal('1e-40'):
            break
        nu = new_nu
    else:
        raise ArithmeticError("scaled mass iteration did not converge")
    _, slope = analytic_slope(n, nu)
    return slope*float(N)**(2/3)


def main():
    rows = []
    with localcontext() as context:
        context.prec = 104
        initial = airy_initial_data()
        for mass in (0, 1, 2, 3, 5):
            prediction = crossover_coefficient(mass, initial)
            alternate = crossover_coefficient(mass, initial, cutoff=16, tail_terms=8)
            assert abs(alternate-prediction) < 5e-8
            with localcontext(context) as copied:
                copied.prec = 210
                errors = [abs(slope_at_scaled_mass(n, mass)-alternate) for n in (64, 256)]
            assert errors[1] < errors[0] and errors[1] < 1e-4
            rows.append(
                f"{mass} & ${alternate.real:.7f}{alternate.imag:+.7f}\\ii$ "
                f"& {errors[0]:.2e} & {errors[1]:.2e} " + r"\\"
            )
            print(f"scaled mass={mass}: Airy={alternate}, errors={errors}, "
                  f"cutoff change={abs(alternate-prediction):.2e}", flush=True)
        assert abs(5*alternate-(0.5+0.5j)) < 0.003
        with localcontext(context) as copied:
            copied.prec = 210
            large_mass = 10*slope_at_scaled_mass(256, 10)
        assert abs(large_mass-(0.5+0.5j)) < 4e-5
        print(f"scaled mass=10: geometric test at n=256 gives {large_mass}")
    Path(__file__).with_name("lovelock_crossover_rows.tex").write_text(
        "\n".join(rows + [r"\hline"]) + "\n", encoding="ascii"
    )


if __name__ == "__main__":
    main()
