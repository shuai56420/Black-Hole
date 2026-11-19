"""Check reflection singularities and the jumps of continued Laplace integrals."""

import cmath
import math
from pathlib import Path

from check_lovelock import log_gamma
from check_lovelock_residues import zero_momentum_residue
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


def scaled_image(j, k, nu, eps, lateral=1, tilt=math.pi/2, cut_tilt=2*math.pi/3):
    """Return epsilon**(2*nu+1) G> / (C_nu/(2*pi)) on the chosen side."""
    n = j + k
    column = j - k
    t = column*TAU + 1j*(n*TAU - eps)
    phase = math.pi * (nu + 1)
    scale = eps ** (2*nu + 1)

    def thermal_denominator(omega):
        theta = TAU*omega - phase
        q = math.exp(-TAU * omega)
        return 1 - q * cmath.exp(1j * theta)

    def thermal_envelope(omega):
        if omega == 0 and nu % 2 == 1:
            # The gamma ratio is a polynomial with a simple zero in
            # each of s_+ and s_-. Cancel the thermal kernel's zero.
            return 8**(nu-1) * math.gamma((1+nu)/2)**4 / (2*TAU**2)
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
        for index in range(s+1):
            difference = s - 2*index
            if s == n and difference == column:
                continue
            L = s*TAU + 1j*(t - difference*TAU)
            angle = -math.copysign(tilt, L.imag) if difference != column else -lateral*cut_tilt
            regular += cmath.exp(-1j*difference*phase) * tilted_laplace(L, nu, angle)

    def singular(u):
        return math.exp(-u) * eps**(2*nu) * envelope(u/eps, nu)

    return (
        scale * regular + cmath.exp(-1j*column*phase)
        * integral_between(singular, eps, 60)
    )


def scaled_wightman(n, nu, eps, side=1, tilt=math.pi/2):
    j, k = (n, 0) if side > 0 else (0, n)
    return scaled_image(j, k, nu, eps, tilt=tilt)


def laplace_jump(a, nu, terms=100):
    """Upper-minus-lower jump of Phi_nu(-a), via the gamma pole residues."""
    if nu == round(nu):
        return 0j
    h = (1 + nu) / 2
    normalization = 2*(1-1j) * 8**nu / math.gamma(-nu)
    pole_sum = 0j
    binomial = 1
    for index in range(terms):
        height = h + index
        ratio = cmath.exp(log_gamma(h + 1j*height) - log_gamma(1-h + 1j*height))
        pole = -2*(1-1j)*height
        pole_sum += binomial * ratio * cmath.exp(a*pole)
        binomial *= (nu + 1 + index) / (1 + index)
    # The closing arc runs clockwise through the left half-plane.
    return -4j*math.pi * (normalization * pole_sum).real


def jump_from_retarded_modes(time, nu, terms=100):
    """Use the positive- and negative-frequency towers of the retarded kernel."""
    h = (1+nu)/2
    retarded = 2*math.fsum(
        (zero_momentum_residue(nu, n)*cmath.exp(-2*(1+1j)*(n+h)*time)).imag
        for n in range(terms)
    )
    vacuum = 2*math.pi/(4**nu*math.gamma(nu)*math.gamma(1+nu))
    return 4j*math.pi*math.sin(math.pi*nu)*retarded/vacuum


def check_mixed_images():
    largest_kernel_error = 0
    largest_mode_error = 0
    for nu in (0.75, 1, 1.25, 1.5):
        for a in (0.8, 2.5):
            direct = tilted_laplace(-a, nu, -2*math.pi/3) - tilted_laplace(-a, nu, 2*math.pi/3)
            residue = laplace_jump(a, nu)
            error = abs(direct - residue)
            largest_kernel_error = max(largest_kernel_error, error)
            mode_error = abs(direct-jump_from_retarded_modes(a, nu))
            largest_mode_error = max(largest_mode_error, mode_error)
            assert error < 1e-7
            assert mode_error < 1e-7
    print(f"Maximum absolute difference for kernel jump: {largest_kernel_error:.2e}")
    print(f"Maximum absolute difference from the real-time retarded sum: {largest_mode_error:.2e}")

    rows = []
    for j, k, nu in ((1, 1, 0.75), (2, 1, 0.75), (2, 1, 1.25), (2, 2, 1.5)):
        n = j + k
        column = j - k
        phase = cmath.exp(-1j*column*math.pi*(nu+1))
        limit = phase * math.gamma(2*nu+1)
        errors = []
        for epsilon in (0.16, 0.08, 0.04):
            right = scaled_image(j, k, nu, epsilon)
            left = scaled_image(j, k, nu, epsilon, lateral=-1)
            errors.append(max(abs(right/limit-1), abs(left/limit-1)))
        assert errors[2] < errors[1] < errors[0] and errors[2] < 5e-4
        expected = epsilon**(2*nu+1) * phase * sum(
            jump_from_retarded_modes((n-s)*TAU-epsilon, nu)
            for s in range(abs(column), n, 2)
        )
        assert abs(right-left-expected) < 2e-9
        alternate = scaled_image(j, k, nu, epsilon, cut_tilt=5*math.pi/8)
        assert abs(alternate-right) < 2e-9
        values = " & ".join(f"{error:.3e}" for error in errors)
        rows.append(f"({j},{k}) & {nu:g} & {values} " + r"\\")
        print(f"(j,k)=({j},{k}), nu={nu:g}: relative errors={values}; "
              f"jump residual={abs(right-left-expected):.2e}")
    Path(__file__).with_name("lovelock_mixed_rows.tex").write_text(
        "\n".join(rows + [r"\hline"]) + "\n", encoding="ascii"
    )


def check_subleading_singularities():
    rows = []
    for j, k, nu in ((2, 0, 2), (3, 0, 2.25), (2, 1, 2.25)):
        leading = math.gamma(2*nu+1)
        phase = cmath.exp(-1j*(j-k)*math.pi*(nu+1))
        envelope_coefficient = 2*nu*(nu**2-1)*(3*nu**2-7)/15
        expected = envelope_coefficient*math.gamma(2*nu-3)
        errors = []
        for epsilon in (0.12, 0.06, 0.03):
            normalized = scaled_image(j, k, nu, epsilon)/phase
            measured = (normalized-leading)/epsilon**4
            errors.append(abs(measured/expected-1))
        assert errors[0] > errors[1] > errors[2]
        assert errors[-1] < 0.001
        cells = " & ".join(f"{error:.3e}" for error in errors)
        rows.append(f"$({j},{k})$ & {nu:g} & {cells} " + r"\\")
        print(f"({j},{k}) at nu={nu:g}: subleading coefficient errors {errors}")
    Path(__file__).with_name("lovelock_subleading_time_rows.tex").write_text(
        "\n".join(rows+[r"\hline"])+"\n", encoding="ascii"
    )

    errors = []
    for time in (0.25, 0.125, 0.08):
        angle = 2*math.pi/3
        direct = tilted_laplace(-time, 1.5, -angle)-tilted_laplace(-time, 1.5, angle)
        from_modes = jump_from_retarded_modes(time, 1.5, terms=600)
        assert abs(direct-from_modes) < 5e-8
        errors.append(abs(direct/(1j*math.pi/8)-1))
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 4e-4
    print(f"nu=1.5: relative errors from the logarithmic jump {errors}")


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
    check_mixed_images()
    check_subleading_singularities()


if __name__ == "__main__":
    main()
