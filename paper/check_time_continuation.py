"""Check primitive reflection singularities using continued Laplace integrals."""

import cmath
import math
from pathlib import Path

from check_lovelock import log_gamma
from check_periods import integrate


TAU = math.pi / 2


def envelope(omega, nu):
    """The analytic continuation of omega**(2*nu) P_nu(omega/2)."""
    h = (1 + nu) / 2
    value = 3 * nu * math.log(2)
    for s in ((1 + 1j)*omega/4, (1 - 1j)*omega/4):
        value += log_gamma(h + s) - log_gamma(1 - h + s)
    return cmath.exp(value)


def integral_between(function, a, b, tol=1e-10):
    return (b - a) * integrate(lambda x: function(a + (b - a) * x), tol)


def tilted_laplace(L, nu, angle):
    """Start at omega=1, then deform to the indicated decaying ray."""
    ray = cmath.exp(1j * angle)
    damping = (L * ray).real
    if damping <= 0:
        raise ValueError("Laplace integral would grow on the chosen ray")
    # Turning the lower-limit segment as well as the tail preserves
    # the original integration cycle.
    def arc(theta):
        omega = cmath.exp(1j * theta)
        return 1j * omega * cmath.exp(-L * omega) * envelope(omega, nu)

    def tail(radius):
        omega = ray * radius
        return ray * cmath.exp(-L * omega) * envelope(omega, nu)

    return integral_between(arc, 0, angle) + integral_between(tail, 1, 1 + 60/damping)


def scaled_wightman(n, nu, eps, side=1, tilt=math.pi/2):
    """Return epsilon**(2*nu+1) G> / (C_nu/(2*pi)), approached vertically."""
    t = side*n*TAU + 1j*(n*TAU - eps)
    phase = math.pi * (nu + 1)
    scale = eps ** (2*nu + 1)

    def thermal_denominator(omega):
        # nu=1 is included as a check. Handle the removable zero at
        # omega=0 directly in the integrand below.
        theta = TAU*omega - phase
        q = math.exp(-TAU * omega)
        return 1 - q * cmath.exp(1j * theta)

    def thermal_envelope(omega):
        if omega == 0 and nu == 1:
            return 1 / (2 * TAU**2)
        return envelope(omega, nu).real / abs(thermal_denominator(omega))**2

    def below_cutoff(omega):
        return thermal_envelope(omega) * (
            cmath.exp(-1j*t*omega) + cmath.exp((1j*t - 2*TAU)*omega)
        )

    def negative_tail(omega):
        return thermal_envelope(omega) * cmath.exp((1j*t - 2*TAU)*omega)

    def positive_tail(omega):
        # Sum all j+k >= n+1 geometrically. Factoring out the decay
        # avoids subtracting nearly equal thermal kernels at large omega.
        eta = cmath.exp(1j * (TAU*omega - phase))
        denominator = thermal_denominator(omega)
        tail = (
            sum(eta**(n+1-2*k) for k in range(n+1)) / denominator
            + eta.conjugate()**(n+1) / abs(denominator)**2
        )
        return (
            envelope(omega, nu) * tail
            * cmath.exp(-((n+1)*TAU + 1j*t)*omega)
        )

    regular = integral_between(below_cutoff, 0, 1)
    regular += integral_between(negative_tail, 1, 1 + 60/(2*TAU + t.imag))
    regular += integral_between(positive_tail, 1, 1 + 60/((n+1)*TAU - t.imag))
    for s in range(n+1):
        for k in range(s+1):
            difference = s - 2*k
            if s == n and difference == side*n:
                continue
            L = s*TAU + 1j*(t - difference*TAU)
            angle = -math.copysign(tilt, L.imag)
            regular += cmath.exp(-1j*difference*phase) * tilted_laplace(L, nu, angle)

    def singular(u):
        return math.exp(-u) * eps**(2*nu) * envelope(u/eps, nu)

    return (
        scale * regular + cmath.exp(-1j*side*n*phase)
        * integral_between(singular, eps, 60)
    )


def main():
    rows = []
    for n, nu in ((1, 0.75), (2, 0.75), (2, 1), (2, 1.25), (3, 1.25)):
        limit = math.gamma(2*nu + 1) * cmath.exp(-1j*n*math.pi*(nu + 1))
        errors = []
        for epsilon in (0.16, 0.08, 0.04):
            scaled = scaled_wightman(n, nu, epsilon)
            errors.append(abs(scaled / limit - 1))
        assert errors[2] < errors[1] < errors[0] and errors[2] < 5e-4
        alternate = scaled_wightman(n, nu, 0.04, tilt=math.pi/3)
        negative = scaled_wightman(n, nu, 0.04, side=-1)
        assert abs(alternate - scaled) < 2e-9
        assert abs(negative - scaled.conjugate()) < 2e-9
        values = " & ".join(f"{error:.3e}" for error in errors)
        rows.append(f"{n} & {nu:g} & {values} " + r"\\")
        print(f"n={n}, nu={nu:g}: relative errors={values}; "
              f"ray change={abs(alternate-scaled):.2e}")
    Path(__file__).with_name("lovelock_time_rows.tex").write_text(
        "\n".join(rows + [r"\hline"]) + "\n", encoding="ascii"
    )


if __name__ == "__main__":
    main()
