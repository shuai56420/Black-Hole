"""Compare the analytic momentum dispersion with full Lovelock modes."""

import cmath
import math
from decimal import Decimal, localcontext
from pathlib import Path

from check_einstein_poles import ComplexDecimal, about_horizon, find_zero, frobenius_at


def analytic_slope(n, nu):
    """Compute d omega / d(k**2) from the continued Jacobi norm."""
    h = (1+nu)/2
    pole = ComplexDecimal(2*(h+n), -2*(h+n))
    alpha = ComplexDecimal(-(h+n), -(h+n))

    polynomial = ComplexDecimal(1)
    term = ComplexDecimal(1)
    for j in range(n):
        term = term * (alpha+nu+n+1+j) / (alpha+1+j) * (Decimal(n-j)/(j+1))
        polynomial = polynomial+term

    beta_ratio = ComplexDecimal(1)
    for j in range(n+1):
        beta_ratio = beta_ratio * (alpha+j) / (alpha+nu+n+1+j)

    # Pfaff's transformation puts the hypergeometric argument at 1/2.
    integral = ComplexDecimal(1)
    term = ComplexDecimal(1)
    c = alpha+nu+2*n+2
    for j in range(2000):
        term = term * (Decimal(n+1+j)*(nu+n+1+j)/(2*(j+1))) / (c+j)
        integral = integral+term
        if abs(complex(term)) <= 1e-65*max(1, abs(complex(integral))):
            break
    else:
        raise ArithmeticError("Jacobi moment did not converge")
    weight = ComplexDecimal(1) + alpha/(alpha+nu+2*n+1)
    slope = polynomial*beta_ratio*integral * (Decimal(2)**(-n)) / (pole*weight)
    return complex(pole), complex(slope)


def heun_operator(nu, momentum_squared, omega, horizon):
    """Polynomials multiplying z**2 A'', z A', and A."""
    mass = (nu*nu-1)/4
    a = [0, 0, 1, 0, -2, 0, 1]
    b = [0, 0, 0, -2, 0, 2]
    c = [-mass, -momentum_squared/4, mass, momentum_squared/4, 0]
    if horizon:
        a, b, c = about_horizon(a)[2:], about_horizon(b, -1)[1:], about_horizon(c)
        frequency_powers = (1, -1)
        start = 0
        exponent = ComplexDecimal(omega.imag/4, -omega.real/4)
    else:
        a, b = a[2:], b[1:]
        frequency_powers = (1,)
        start = 1
        exponent = ComplexDecimal((1+nu)/2)
    c = [ComplexDecimal(value) for value in c]
    frequency = omega*omega/4
    for index, factor in enumerate(frequency_powers, start=start):
        c[index] = c[index]+frequency*factor
    return tuple(zip(a, b, c)), exponent


def numerical_pole(nu, momentum_squared, guess, point, terms):
    def mismatch(omega):
        operator, exponent = heun_operator(nu, momentum_squared, omega, True)
        _, horizon_derivative = frobenius_at(operator, exponent, 1-point, terms)
        operator, exponent = heun_operator(nu, momentum_squared, omega, False)
        _, boundary_derivative = frobenius_at(operator, exponent, point, terms)
        return horizon_derivative*(-1)-boundary_derivative

    return find_zero(mismatch, guess)


def main():
    rows = []
    with localcontext() as context:
        context.prec = 95
        for nu, n in ((Decimal('0.75'), 0), (Decimal('0.75'), 3), (Decimal(1), 5),
                      (Decimal('1.25'), 0), (Decimal('1.25'), 8)):
            unperturbed, expected = analytic_slope(n, nu)
            slopes = []
            for step, point, terms in ((Decimal('0.02'), Decimal('0.54'), 280),
                                       (Decimal('0.01'), Decimal('0.61'), 420)):
                poles = [
                    numerical_pole(nu, sign*step, unperturbed, point, terms)
                    for sign in (1, -1)
                ]
                slopes.append((poles[0]-poles[1])/(2*float(step)))
            extrapolated = (4*slopes[1]-slopes[0])/3
            error = abs(extrapolated-expected)
            assert abs(slopes[1]-expected) < abs(slopes[0]-expected)
            assert error < 3e-10
            rows.append(
                f"{float(nu):g} & {n} & "
                f"${expected.real:.8f}{expected.imag:+.8f}\\ii$ & {error:.2e} "
                + r"\\"
            )
            print(f"nu={nu}, n={n}: analytic slope={expected}, "
                  f"numerical={extrapolated}, residual={error:.2e}", flush=True)
    check_airy_limit()
    Path(__file__).with_name("lovelock_momentum_rows.tex").write_text(
        "\n".join(rows + [r"\hline"]) + "\n", encoding="ascii"
    )


def check_airy_limit():
    optical = math.pi*(1+1j)/4
    airy_coefficient = (
        1j*math.gamma(1/3)**2*cmath.exp(-1j*math.pi/3)
        / (8*optical*3**(1/3)*(1-1j)**(2/3))
    )
    rows = []
    with localcontext() as context:
        context.prec = 140
        for nu in (Decimal('0.75'), Decimal('1.25'), Decimal(2)):
            errors = []
            for n in (16, 64, 256):
                _, slope = analytic_slope(n, nu)
                scaled = slope*float(n+(1+nu)/2)**(2/3)
                errors.append(abs(scaled-airy_coefficient))
                print(f"nu={nu}, n={n}: scaled slope={scaled}, Airy error={errors[-1]:.2e}")
            assert errors[2] < errors[1] < errors[0] and errors[2] < 1.2e-4
            values = " & ".join(f"{error:.3e}" for error in errors)
            rows.append(f"{float(nu):g} & {values} " + r"\\")
    Path(__file__).with_name("lovelock_airy_rows.tex").write_text(
        "\n".join(rows + [r"\hline"]) + "\n", encoding="ascii"
    )


if __name__ == "__main__":
    main()
