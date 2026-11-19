"""Check the nonlinear transverse geodesic action and its WKB limit."""

import cmath
import math
from pathlib import Path

from check_lovelock_scaled_momentum import connection_multiplier
from check_periods import integrate


def optical_limit(scaled_momentum):
    """Evaluate the elliptic period, continued from zero momentum."""
    outer = (1+cmath.sqrt(1-4*scaled_momentum))/2
    inner = scaled_momentum/outer

    def integrand(t):
        return math.pi/cmath.sqrt(outer-inner*math.cos(math.pi*t/2)**2)

    return integrate(integrand)


def finite_energy_period(energy, scaled_momentum):
    """Subtract the imaginary-ray tail from the positive-ray period."""
    epsilon = energy**(-4)
    a = (1+math.sqrt(1-4*scaled_momentum))/2

    def root(initial):
        u = initial
        for _ in range(12):
            polynomial = u*u*(1+epsilon*(u-scaled_momentum))-u+scaled_momentum
            derivative = u*(2+epsilon*(3*u-2*scaled_momentum))-1
            change = polynomial/derivative
            u -= change
            if abs(change) <= 4e-16*abs(u):
                return u
        raise ArithmeticError("finite-energy turning point did not converge")

    def doubled_leg(turning_point):
        sign = math.copysign(1, turning_point)

        def integrand(t):
            cosine = math.cos(math.pi*t/2)
            square = cosine*cosine
            denominator = (epsilon*turning_point**2
                           + (1-scaled_momentum/turning_point)*square
                           - scaled_momentum/turning_point*square*square)
            return sign*math.pi*cosine/math.sqrt(denominator)

        return integrate(integrand).real

    real_root = root(a)
    imaginary_root = root(-1/epsilon+scaled_momentum-1)
    return doubled_leg(real_root)-doubled_leg(imaginary_root)


def main():
    directory = Path(__file__).parent
    geo_rows = []
    for eta in (0.0625, 0.16, 0.24):
        expected = optical_limit(eta)
        errors = [abs(finite_energy_period(energy, eta)/expected-1)
                  for energy in (8, 32, 128)]
        assert errors[0] > errors[1] > errors[2]
        assert errors[-1] < 3e-7
        cells = " & ".join(f"{error:.2e}" for error in errors)
        geo_rows.append(f"{eta:g} & {expected.real:.7f} & {cells} " + r"\\")
        print(f"eta={eta}: Jopt={expected.real:.7f}, finite-energy errors={errors}")
    directory.joinpath("geodesic_momentum_rows.tex").write_text(
        "\n".join(geo_rows+[r"\hline"])+"\n", encoding="ascii"
    )

    connection_rows = []
    for strength in (0.12, 0.24):
        eta = -1j*strength
        action = eta*integrate(lambda t: optical_limit(t*eta))
        errors = []
        for chi in (3, 4, 5):
            shift = cmath.exp(-1j*math.pi/3)*chi**2/(4*2**(2/3))
            coupling = -shift**2*eta
            amplitude, residual = connection_multiplier(coupling, 38, .003, 38, shift)
            refined, refined_residual = connection_multiplier(coupling, 43, .0012, 44, shift)
            assert max(residual, refined_residual) < 0.0003
            assert abs(amplitude/refined-1) < 1e-6
            log_connection = cmath.log(refined)
            predicted = -shift**1.5*action
            error = abs(log_connection/predicted-1)
            errors.append(error)
            print(f"eta={eta}, chi={chi}: geometric action={predicted}, "
                  f"Airy log={log_connection}, relative error={error:.2e}")
        assert errors[0] > errors[1] > errors[2]
        assert errors[-1] < 0.003
        cells = " & ".join(f"{error:.2e}" for error in errors)
        connection_rows.append(f"$-{strength:g}\\ii$ & {cells} " + r"\\")
    directory.joinpath("lovelock_geometric_momentum_rows.tex").write_text(
        "\n".join(connection_rows+[r"\hline"])+"\n", encoding="ascii"
    )


if __name__ == "__main__":
    main()
