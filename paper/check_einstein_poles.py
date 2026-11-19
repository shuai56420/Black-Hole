"""Check Einstein-brane quasinormal poles and reflection harmonics."""

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


def equation(d, nu, omega, horizon, source):
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
        exponent = ComplexDecimal(Decimal('0.5') + (-nu if source else nu))
        degree_shift = 2
    c = [ComplexDecimal(value) for value in c]
    for index, value in enumerate(powers, start=degree_shift):
        c[index] = c[index] + omega*omega*value
    return tuple(zip(a, b, c)), exponent


def endpoint_series(d, nu, omega, point, terms, horizon=False, source=False):
    """Return the regular part of a Frobenius solution and its log derivative."""
    operator, exponent = equation(d, nu, omega, horizon, source)
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
    return value, derivative / value / (-z if horizon else z)


def find_pole(d, nu, guess, point, terms):
    def mismatch(omega):
        _, horizon_derivative = endpoint_series(d, nu, omega, point, terms, horizon=True)
        _, normal_derivative = endpoint_series(d, nu, omega, point, terms)
        return horizon_derivative - normal_derivative

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


def optical_distance(d):
    theta = math.pi/d
    return theta*cmath.exp(1j*theta)/math.sin(theta)


def leading_pole(d, nu, n):
    return (math.pi*(n+(nu-1)/2) + 0.5j*math.log(2))/optical_distance(d)


def stokes_slope(power):
    moment = 2**(power-3)*math.gamma(power/2)**4 / math.gamma(power)
    b_prime = -math.tan(math.pi*power/2)*moment/math.pi
    return 2*b_prime*(math.cos(math.pi*power)-1)


def tuned_reflection_coefficient(d, nu):
    """Coefficient of omega**(-2*p) in the tuned R(omega) + 2."""
    power = 2*d/(d-1)
    a = (d-1)/(2*d-1)
    b = (d-1)/(3*d-1)
    coefficient = nu**2 - 0.25 + (d-1)**2*(3*a*a-2*b)/4
    return stokes_slope(power)*coefficient*(d-1)**(power-2)*cmath.exp(-0.5j*math.pi*power)


def scaled_shift(d, nu, n, pole, tuned):
    p = d/(d-1)
    leading = leading_pole(d, nu, n)
    if tuned:
        power = 2*p
        norm = -0.25j*tuned_reflection_coefficient(d, nu)
    else:
        power = p
        norm = 0.25j*stokes_slope(p)*(d-1)**(p-2)*cmath.exp(-0.5j*math.pi*p)
    return (pole-leading)*optical_distance(d)*leading**power/norm


def vacuum_normalized_response(d, nu, frequency, point, terms):
    """Compute B/A of the ingoing solution in the Frobenius basis at u=0."""
    omega = ComplexDecimal(str(frequency.real), str(frequency.imag))
    _, horizon_derivative = endpoint_series(d, nu, omega, point, terms, horizon=True)
    normal, normal_derivative = endpoint_series(d, nu, omega, point, terms)
    source, source_derivative = endpoint_series(d, nu, omega, point, terms, source=True)
    ratio = (source / normal * (horizon_derivative - source_derivative)
             / (normal_derivative - horizon_derivative) * point**(-2*nu))
    dimension = float(nu)
    vacuum = (
        -cmath.exp(-1j*math.pi*dimension)
        * math.gamma(1-dimension)/math.gamma(1+dimension)
        * (frequency/2)**(2*dimension)
    )
    return complex(ratio) / vacuum


def reflection_harmonics(d, nu, n, radius, samples, point, terms):
    """Project the exact response over one phase period on the tilted line."""
    optical = optical_distance(d)
    dimension = float(nu)
    initial_pole = leading_pole(d, dimension, n)
    r2 = tuned_reflection_coefficient(d, dimension)
    power = 2*d/(d-1)
    phase_factor = 1-cmath.exp(2j*math.pi*dimension)
    leading = [0j, 0j]
    correction = [0j, 0j]
    for index in range(samples):
        theta = math.tau*index/samples
        frequency = initial_pole + (theta-1j*math.log(radius))/(2*optical)
        symbol = radius*cmath.exp(1j*theta)
        response = vacuum_normalized_response(d, nu, frequency, point, terms)
        reduced = (response-1)/phase_factor
        subtracted = (reduced-symbol/(1-symbol))/(-r2*frequency**(-power))
        for slot, harmonic in enumerate((2, 3)):
            weight = symbol**(-harmonic)/samples
            leading[slot] += weight*reduced
            correction[slot] += weight*subtracted*(2/harmonic)
    return leading, correction


def check_reflection_harmonics(d, nu):
    rows = []
    for n in (5, 15, 30):
        leading, correction = reflection_harmonics(d, nu, n, 0.4, 48, Decimal('0.53'), 420)
        with localcontext() as check_context:
            check_context.prec += 25
            check_leading, check_correction = reflection_harmonics(
                d, nu, n, 0.4, 72, Decimal('0.61'), 540
            )
        leading_change = max(abs(a-b) for a, b in zip(leading, check_leading))
        correction_change = max(abs(a-b) for a, b in zip(correction, check_correction))
        assert leading_change < 2e-5
        assert correction_change < 5e-3
        print(f"n={n} leading harmonics={check_leading} "
              f"normalized corrections={check_correction}", flush=True)
        print(f"  refinement changes: {leading_change:.2e}, {correction_change:.2e}", flush=True)
        leading_errors = [abs(value-1) for value in check_leading]
        correction_errors = [abs(value-1) for value in check_correction]
        if n == 30:
            assert max(leading_errors) < 6e-6
            assert max(correction_errors) < 0.007
        values = " & ".join(f"{error:.3e}" for error in leading_errors + correction_errors)
        rows.append(f"{n} & {values} " + r"\\")
    Path(__file__).with_name("einstein_harmonic_rows.tex").write_text(
        "\n".join(rows + [r"\hline"]) + "\n", encoding="ascii"
    )


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
        check_reflection_harmonics(d, tuned_nu)
    Path(__file__).with_name("einstein_pole_rows.tex").write_text(
        "\n".join(rows + [r"\hline"]) + "\n", encoding="ascii"
    )


if __name__ == "__main__":
    main()
