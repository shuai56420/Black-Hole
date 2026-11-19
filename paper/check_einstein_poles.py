"""Match the ingoing and normalizable scalar series in Schwarzschild–AdS."""

import cmath
import math
from decimal import Decimal, localcontext
from pathlib import Path


class ComplexDecimal:
    """Complex arithmetic for the endpoint series and secant iteration."""

    def __init__(self, value=0, imag=0):
        self.real = Decimal(value)
        self.imag = Decimal(imag)

    def __add__(self, other):
        if isinstance(other, ComplexDecimal):
            return ComplexDecimal(self.real + other.real, self.imag + other.imag)
        return ComplexDecimal(self.real + other, self.imag)

    def __sub__(self, other):
        if isinstance(other, ComplexDecimal):
            return ComplexDecimal(self.real - other.real, self.imag - other.imag)
        return ComplexDecimal(self.real - other, self.imag)

    def __mul__(self, other):
        if isinstance(other, ComplexDecimal):
            return ComplexDecimal(self.real*other.real - self.imag*other.imag,
                                  self.real*other.imag + self.imag*other.real)
        return ComplexDecimal(self.real*other, self.imag*other)

    def __truediv__(self, other):
        if isinstance(other, ComplexDecimal):
            square = other.real**2 + other.imag**2
            return ComplexDecimal((self.real*other.real + self.imag*other.imag)/square,
                                  (self.imag*other.real - self.real*other.imag)/square)
        return ComplexDecimal(self.real/other, self.imag/other)

    def __complex__(self):
        return complex(float(self.real), float(self.imag))


def about_horizon(coefficients, sign=1):
    """Coefficients under u = 1-z, with a sign for a first derivative."""
    return [
        sign * (-1)**k * sum(math.comb(j, k)*coefficients[j]
                             for j in range(k, len(coefficients)))
        for k in range(len(coefficients))
    ]


def equation(d, nu, omega, horizon):
    """Polynomials A, B, C multiplying z**2 psi'', z psi', psi."""
    potential = nu*nu - Decimal('0.25')
    geometric = Decimal((d-1)**2) / 4
    # Before shifting the horizon, use u**2 times the scalar equation.
    a = [0] * (2*d+3)
    a[2], a[d+2], a[2*d+2] = 1, -2, 1
    b = [0] * (2*d+2)
    b[d+1], b[2*d+1] = -d, d
    c = [0] * (2*d+1)
    c[0], c[d], c[2*d] = -potential, potential-geometric, geometric
    if horizon:
        a, b, c = about_horizon(a)[2:], about_horizon(b, -1)[1:], about_horizon(c)
        powers = (1, -2, 1)
        exponent = ComplexDecimal(omega.imag/d, -omega.real/d)
        degree_shift = 0
    else:
        a, b = a[2:], b[1:]
        powers = (1,)
        exponent = ComplexDecimal(nu + Decimal('0.5'))
        degree_shift = 2
    c = [ComplexDecimal(value) for value in c]
    for index, value in enumerate(powers, start=degree_shift):
        c[index] = c[index] + omega*omega*value
    return tuple(zip(a, b, c)), exponent


def logarithmic_derivative(d, nu, omega, point, terms, horizon):
    """Evaluate a Frobenius series after stripping its power from the value."""
    operator, exponent = equation(d, nu, omega, horizon)
    coefficients = [ComplexDecimal(1)]
    derivatives = [exponent + n for n in range(terms+1)]
    second = [degree*(degree-1) for degree in derivatives]
    first_coefficients = [derivatives[0]]
    a0, b0, c0 = operator[0]
    for n in range(1, terms+1):
        total = ComplexDecimal()
        for k in range(1, min(n+1, len(operator))):
            a, b, c = operator[k]
            if a == 0 and b == 0 and c.real == 0 and c.imag == 0:
                continue
            total = total + (second[n-k]*a + derivatives[n-k]*b + c)*coefficients[n-k]
        pivot = second[n]*a0 + derivatives[n]*b0 + c0
        coefficient = (total / pivot)*(-1)
        coefficients.append(coefficient)
        first_coefficients.append(coefficient*derivatives[n])

    z = 1-point if horizon else point
    value = coefficients[-1]
    derivative = first_coefficients[-1]
    for coefficient, first in zip(reversed(coefficients[:-1]), reversed(first_coefficients[:-1])):
        value = value*z + coefficient
        derivative = derivative*z + first
    return derivative / value / (-z if horizon else z)


def find_pole(d, nu, guess, point, terms):
    def mismatch(omega):
        return (logarithmic_derivative(d, nu, omega, point, terms, True)
                - logarithmic_derivative(d, nu, omega, point, terms, False))

    left = ComplexDecimal(str(guess.real), str(guess.imag))
    right = left + Decimal('0.015')
    f_left = mismatch(left)
    for _ in range(20):
        f_right = mismatch(right)
        delta = f_right*(right-left) / (f_right-f_left)
        if abs(complex(delta)) < 1e-16 and abs(complex(f_right)) < 1e-14:
            return complex(right)
        left, f_left = right, f_right
        right = right-delta
    raise ArithmeticError(f"Pole root did not converge near {guess}")


def leading_pole(d, nu, n):
    theta = math.pi/d
    optical = theta*cmath.exp(1j*theta)/math.sin(theta)
    return (math.pi*(n+(nu-1)/2) + 0.5j*math.log(2))/optical


def stokes_slope(power):
    moment = 2**(power-3)*math.gamma(power/2)**4 / math.gamma(power)
    b_prime = -math.tan(math.pi*power/2)*moment/math.pi
    return 2*b_prime*(math.cos(math.pi*power)-1)


def scaled_shift(d, nu, n, pole, tuned):
    p = d/(d-1)
    theta = math.pi/d
    optical = theta*cmath.exp(1j*theta)/math.sin(theta)
    leading = leading_pole(d, nu, n)
    if tuned:
        # The first Puiseux term vanishes. The coefficient of r**2
        # in V + 1/(4*(x-x_s)**2) then fixes the next term.
        a = (d-1)/(2*d-1)
        b = (d-1)/(3*d-1)
        coefficient = nu**2 - 0.25 + (d-1)**2*(3*a*a-2*b)/4
        power = 2*p
        norm = -0.25j*stokes_slope(power)*coefficient
    else:
        power = p
        norm = 0.25j*stokes_slope(power)
    norm *= (d-1)**(power-2)*cmath.exp(-0.5j*math.pi*power)
    return (pole-leading)*optical*leading**power/norm


def main():
    rows = []
    d = 6
    with localcontext() as ctx:
        ctx.prec = 95
        tuned_nu = Decimal(3) / Decimal(11).sqrt()
        for nu in (Decimal('0.5'), tuned_nu, Decimal('1.25')):
            tuned = nu == tuned_nu
            expected = 1.0 if tuned else float(nu*nu - Decimal(d*d)/(4*(2*d-1)))
            errors = []
            for n in (5, 15, 30):
                guess = leading_pole(d, float(nu), n)
                pole = find_pole(d, nu, guess, Decimal('0.53'), 420)
                with localcontext(ctx) as check_context:
                    check_context.prec += 25
                    check = find_pole(d, nu, pole, Decimal('0.61'), 540)
                assert abs(pole-check) < 1e-10
                shift = scaled_shift(d, float(nu), n, pole, tuned)
                errors.append(abs(shift-expected))
                print(f"nu={float(nu):.9f} n={n} pole={pole} "
                      f"scaled shift={shift} expected={expected}", flush=True)
                rows.append(
                    f"{float(nu):.6f} & {n} & "
                    f"${pole.real:.6f}{pole.imag:+.6f}\\ii$ & "
                    f"${shift.real:.6f}{shift.imag:+.6f}\\ii$ & {expected:.6f} "
                    + r"\\"
                )
            assert errors[2] < errors[1] < errors[0]
            assert errors[2] < (0.007 if tuned else 0.002)
    Path(__file__).with_name("einstein_pole_rows.tex").write_text(
        "\n".join(rows + [r"\hline"]) + "\n", encoding="ascii"
    )


if __name__ == "__main__":
    main()
