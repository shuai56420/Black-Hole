"""Check the first momentum correction to the Lovelock retarded response."""

import cmath
import math
from decimal import Decimal, localcontext
from pathlib import Path

from check_einstein_poles import ComplexDecimal, frobenius_at
from check_lovelock import log_gamma
from check_lovelock_momentum import analytic_slope, heun_operator


def hypergeometric_half(a, b, c, count):
    """Return the coefficients of F(a,b;c;y/2) truncated at count terms."""
    values = [1+0j]
    for j in range(count-1):
        values.append(values[-1]*(a+j)*(b+j)/(2*(c+j)*(j+1)))
    return values


def product(left, right):
    """Multiply two series, discarding terms outside the original cutoff."""
    limit = len(left)
    values = [0j]*limit
    for i, first in enumerate(left):
        for j in range(limit-i):
            values[i+j] += first*right[j]
    return values


def moment(left, right, weight, exponent):
    """Analytic continuation of a product integral over the first half interval."""
    integrand = product(product(left, right), weight)
    return 0.5**(exponent+1)*sum(
        coefficient/(exponent+1+j) for j, coefficient in enumerate(integrand)
    )


def boundary_basis(omega, nu, count):
    """Hypergeometric coefficients matched between the horizon and boundary."""
    h = (1+nu)/2
    a, b = h-(1+1j)*omega/4, h+(1-1j)*omega/4
    c = 1-1j*omega/2
    ingoing = hypergeometric_half(a, b, c, count)
    source = hypergeometric_half(c-a, c-b, 1-nu, count)
    normal = hypergeometric_half(a, b, 1+nu, count)

    def midpoint(coefficients):
        return sum(coefficients), 2*sum(j*value for j, value in enumerate(coefficients))

    H, dH = midpoint(ingoing)
    S, dS = midpoint(source)
    V, dV = midpoint(normal)
    S, dS = 2**nu*S, 2**nu*(2*nu*S-dS)
    dV = -dV
    wronskian = S*dV-dS*V
    return ingoing, source, normal, (H*dV-dH*V)/wronskian, (S*dH-dS*H)/wronskian


def analytic_response_slope(omega, nu, count=100):
    ingoing, source, normal, s, v = boundary_basis(omega, nu, count)
    alpha = -1j*omega/2

    def weight(exponent, pole, initial):
        values = [complex(initial)]
        binomial = complex(initial)
        for j in range(count-1):
            binomial *= (j-exponent)/(2*(j+1))
            values.append(binomial+pole*values[-1])
        return values

    horizon = moment(ingoing, ingoing, weight(nu, -0.5, 1), alpha)
    boundary_weight = weight(alpha, 0.25, 0.5)
    boundary = (
        s*s*moment(source, source, boundary_weight, -nu)
        + 2*s*v*moment(source, normal, boundary_weight, 0)
        + v*v*moment(normal, normal, boundary_weight, nu)
    )
    return -2**(nu-1)*(horizon+boundary)/(2*nu*s*s)


def numerical_response(frequency, nu, momentum_squared, point, terms):
    omega = ComplexDecimal(str(frequency.real), str(frequency.imag))
    operator, exponent = heun_operator(nu, momentum_squared, omega, True)
    _, horizon_derivative = frobenius_at(operator, exponent, 1-point, terms)
    horizon_derivative = horizon_derivative*(-1)
    operator, exponent = heun_operator(nu, momentum_squared, omega, False)
    vev, vev_derivative = frobenius_at(operator, exponent, point, terms)
    operator, exponent = heun_operator(nu, momentum_squared, omega, False, source=True)
    source, source_derivative = frobenius_at(operator, exponent, point, terms)
    return complex(
        source / vev * (horizon_derivative-source_derivative)
        / (vev_derivative-horizon_derivative) * point**(-nu)
    )


def numerical_response_slope(frequency, nu):
    differences = []
    for step, point, terms in ((Decimal('0.006'), Decimal('0.53'), 180),
                               (Decimal('0.003'), Decimal('0.61'), 280)):
        plus = numerical_response(frequency, nu, step, point, terms)
        minus = numerical_response(frequency, nu, -step, point, terms)
        differences.append((plus-minus)/(2*float(step)))
    return (4*differences[1]-differences[0])/3


def zero_momentum_residue(nu, n):
    h = (1+nu)/2
    N = n+h
    log_weight = (
        math.lgamma(n+nu+1)-math.lgamma(n+1)-math.lgamma(nu+1)-math.lgamma(nu)
        + log_gamma(h-1j*N)-log_gamma(1-h-1j*N)
    )
    return -4*2**nu*cmath.exp(log_weight)/(1+1j)


def laurent_pair(function, pole, radius, samples):
    double = simple = 0j
    for j in range(samples):
        offset = radius*cmath.exp(1j*math.tau*j/samples)
        value = function(pole+offset)*offset/samples
        simple += value
        double += value*offset
    return double, simple


def check_retarded_sum():
    """Compare the tuned pole sum and the transform of the exact spectrum."""
    tau = math.pi/2
    for time in (0.7, 1.3, 2.1):
        q = cmath.exp(-2*(1+1j)*time)
        poles = 8*((1+1j)*q*(1+q)/(1-q)**3).imag

        def commutator(decay):
            return (decay+1j*time)**(-3)-(decay-1j*time)**(-3)

        spectral = commutator(0)
        for n in range(1, 8193):
            decay = n*tau*(1+1j)
            spectral += commutator(decay)+commutator(decay.conjugate())
        transformed = 0.5j*spectral
        error = abs(poles-transformed)
        assert error < 1e-10
        print(f"nu=1, t={time}: pole sum={poles:.12f}, spectral error={error:.2e}")


def main():
    rows = []
    with localcontext() as context:
        context.prec = 90
        for nu, n in ((Decimal('0.5'), 0), (Decimal('0.75'), 2), (Decimal('1.25'), 3)):
            pole, pole_shift = analytic_slope(n, nu)
            scalar_dimension = float(nu)
            predicted_double, predicted_simple = laurent_pair(
                lambda omega: analytic_response_slope(omega, scalar_dimension), pole, 0.3, 20
            )
            direct_double, direct_simple = laurent_pair(
                lambda omega: numerical_response_slope(omega, nu), pole, 0.3, 20
            )
            _, alternate = laurent_pair(
                lambda omega: analytic_response_slope(omega, scalar_dimension, count=120),
                pole, 0.21, 30
            )
            double_error = abs(predicted_double-zero_momentum_residue(scalar_dimension, n)*pole_shift)
            direct_error = max(abs(direct_double-predicted_double), abs(direct_simple-predicted_simple))
            assert abs(alternate-predicted_simple) < 1e-7
            assert double_error < 1e-7
            assert direct_error < 3e-9
            rows.append(
                f"{nu:g} & {n} & "
                f"${predicted_simple.real:.7f}{predicted_simple.imag:+.7f}\\ii$ "
                f"& {double_error:.2e} & {direct_error:.2e} " + r"\\"
            )
            print(f"nu={nu:g}, n={n}: residue derivative={predicted_simple}, "
                  f"shift check={double_error:.2e}, direct check={direct_error:.2e}", flush=True)
        for nu, frequency in ((Decimal('0.75'), 1.7+0.4j), (Decimal('1.25'), 2.2-0.5j)):
            exact = analytic_response_slope(frequency, float(nu))
            direct = numerical_response_slope(frequency, nu)
            error = abs(exact-direct)
            assert error < 1e-10
            print(f"nu={nu:g}, omega={frequency}: response derivative error={error:.2e}")
    check_retarded_sum()
    Path(__file__).with_name("lovelock_residue_rows.tex").write_text(
        "\n".join(rows + [r"\hline"]) + "\n", encoding="ascii"
    )


if __name__ == "__main__":
    main()
