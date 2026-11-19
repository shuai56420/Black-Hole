"""Check the transverse optical action with both boundary endpoints fixed."""

import cmath
import math
from pathlib import Path

from check_periods import (
    geometric_sum,
    integrate,
    optical_cycle_quadrature,
    outer_turning_point,
    reference_path,
    transverse_reference,
)


def critical_momentum(d):
    m = d-2
    return (2/d)**(2/m)*m/d


def turning_root(equation, initial):
    root = initial
    for _ in range(20):
        value, derivative = equation(root)
        change = value/derivative
        root -= change
        if abs(change) < 3e-15:
            return root
    raise ArithmeticError("turning point did not converge")


def limiting_period(d, eta):
    m = d-2
    root = turning_root(lambda x: (x**d-x*x+eta, d*x**(d-1)-2*x), 1.0)
    strength = eta/root**d

    def integrand(s):
        t = 1-s*s
        denominator = geometric_sum(t, m)-strength*t**m*(1+t)
        return 4/(root*math.sqrt(denominator))

    return integrate(integrand)


def repeated_geodesic(d, bounces, energy, eta):
    """Return the renormalized length, lifted time and transverse period."""
    m = d-2
    scale = energy**(-2/m)
    momentum_squared = eta*scale**2
    inner_epsilon = (scale/energy)**2
    outer_epsilon = energy**(-d)
    outer_momentum = momentum_squared/energy**2

    def inner_equation(x):
        value = 1-x**(-m)+eta*x**(-d)+inner_epsilon*(x*x-eta)
        derivative = m*x**(-m-1)-d*eta*x**(-d-1)+2*inner_epsilon*x
        return value, derivative

    def outer_equation(x):
        value = 1+x*x-outer_momentum-outer_epsilon*(x**(-m)-outer_momentum*x**(-d))
        derivative = 2*x+outer_epsilon*(m*x**(-m-1)-d*outer_momentum*x**(-d-1))
        return value, derivative

    inner = scale*turning_root(inner_equation, 1)
    outer = energy*turning_root(outer_equation, 1j)
    radius = 2*abs(energy)
    angle = cmath.phase(outer)
    inner_span = 0.8-inner
    outer_span = radius/abs(outer)-1

    def radial_factor(inverse_radius):
        return ((energy**2-momentum_squared)*inverse_radius**2
                - inverse_radius**d+momentum_squared*inverse_radius**(d+2))

    def cap_factor(root, y):
        inverse_y = 1/y
        return y+1+root**(-d)*inverse_y*(
            geometric_sum(inverse_y, m)-momentum_squared/root**2*geometric_sum(inverse_y, d)
        )

    def positive_cap(s):
        r = inner+inner_span*s*s
        return r, 2*cmath.sqrt(inner_span/(inner*cap_factor(inner, r/inner)))

    def horizon_detour(s):
        offset = 0.2*cmath.exp(1j*math.pi*(1-s))
        r = 1+offset
        return r, -1j*math.pi*offset/(r*cmath.sqrt(1+radial_factor(1/r)))

    def exterior_leg(s):
        r = 1.2+(radius-1.2)*s
        return r, (radius-1.2)/(r*cmath.sqrt(1+radial_factor(1/r)))

    def exterior_arc(s):
        r = radius*cmath.exp(1j*angle*s)
        return r, 1j*angle/cmath.sqrt(1+radial_factor(1/r))

    def complex_cap(s):
        y = 1+outer_span*s*s
        return outer*y, -2*cmath.sqrt(outer_span/cap_factor(outer, y))

    paths = (positive_cap, horizon_detour, exterior_leg, exterior_arc, complex_cap)
    result = []
    for kind in ("length", "time", "transverse"):
        def integrand(path, s):
            r, length = path(s)
            if kind == "time":
                return energy*length/(r*r-r**(-m))
            if kind == "transverse":
                return length/r**2
            return length

        def tail(t):
            if t == 0:
                return 0
            inverse_radius = t/1.2
            offset = radial_factor(inverse_radius)
            factor = cmath.sqrt(1+offset)
            if kind == "time":
                return energy*inverse_radius**2/(t*factor*(1-inverse_radius**d))
            if kind == "transverse":
                return inverse_radius**2/(t*factor)
            return -offset/(t*factor*(1+factor))

        legs = [integrate(lambda s: integrand(path, s)) for path in paths]
        reference = 2*(sum(legs[:2])+integrate(tail))
        if kind == "length":
            reference -= 2*math.log(1.2)
        optical = 2*sum(legs)
        result.append(reference+(bounces-1)*optical)
    return tuple(result)


def length_at_endpoints(d, bounces, energy, eta, target_time, target_space):
    def data(energy, scaled_momentum):
        length, time, period = repeated_geodesic(d, bounces, energy, scaled_momentum**2)
        space = scaled_momentum*energy**(-2/(d-2))*period
        return length, time, space

    scaled_momentum = math.sqrt(eta)
    for _ in range(8):
        length, time, space = data(energy, scaled_momentum)
        error_time = time-target_time
        error_space = space-target_space
        if max(abs(error_time), abs(error_space)) < 4e-13:
            return length
        step_energy = abs(energy)*1e-5
        step_momentum = 1e-5
        _, time_energy, space_energy = data(energy+step_energy, scaled_momentum)
        _, time_momentum, space_momentum = data(energy, scaled_momentum+step_momentum)
        time_by_energy = (time_energy-time)/step_energy
        time_by_momentum = (time_momentum-time)/step_momentum
        space_by_energy = (space_energy-space)/step_energy
        space_by_momentum = (space_momentum-space)/step_momentum
        determinant = time_by_energy*space_by_momentum-time_by_momentum*space_by_energy
        energy -= (space_by_momentum*error_time-time_by_momentum*error_space)/determinant
        scaled_momentum -= (time_by_energy*error_space-space_by_energy*error_time)/determinant
    raise ArithmeticError("fixed-endpoint solve did not converge")


def main():
    d, energy = 6, 16
    reference = (*reference_path(d, energy), transverse_reference(d, energy))
    cycle = optical_cycle_quadrature(d, energy, outer_turning_point(d, energy), 2*energy)
    original = [base+2*period for base, period in zip(reference, cycle)]
    direct = repeated_geodesic(d, 3, energy, 0)
    assert max(abs(a-b) for a, b in zip(original, direct)) < 5e-11

    rows = []
    for d, bounces in ((5, 2), (6, 3), (8, 3)):
        m = d-2
        q = d/m
        zero = limiting_period(d, 0)
        slope = -math.gamma(1+3/m)*math.gamma(-0.5)/(m*math.gamma(0.5+3/m))
        numerical_slope = (limiting_period(d, 1e-5)-limiting_period(d, -1e-5))/(2e-5)
        assert abs(numerical_slope/slope-1) < 2e-8
        assert abs(zero-2/m*math.gamma(1/m)*math.gamma(0.5)/math.gamma(1/m+0.5)) < 1e-11

        for fraction in (0.5, 0.85):
            eta = fraction*critical_momentum(d)
            period = limiting_period(d, eta)
            action = eta*integrate(lambda t: limiting_period(d, t*eta))
            length_coefficient = bounces*(eta*period-action/2)
            length_errors, action_errors, time_errors = [], [], []
            for energy in (8, 32, 128):
                base_length, time, _ = repeated_geodesic(d, bounces, energy, 0)
                moved_length, moved_time, moved_period = repeated_geodesic(d, bounces, energy, eta)
                time_shift = moved_time-time
                action_shift = moved_length-base_length+energy*time_shift
                action_shift -= eta*energy**(-4/m)*moved_period
                action_errors.append(abs(energy**q*action_shift/(-bounces*action/2)-1))
                predicted_time = bounces*(q*action-4*eta*period/m)/2
                time_errors.append(abs(energy**(q+1)*time_shift-predicted_time))

                space = bounces*math.sqrt(eta)*period/energy
                length = length_at_endpoints(d, bounces, energy, eta, time, space)
                error = abs(energy**q*(length-base_length)/length_coefficient-1)
                length_errors.append(error)
                print(f"d={d}, n={bounces}, eta/eta_c={fraction}, E={energy}: "
                      f"length error={error:.3e}, action error={action_errors[-1]:.3e}, "
                      f"scaled time error={time_errors[-1]:.3e}", flush=True)
            for errors, limit in ((length_errors, .0003), (action_errors, .0005), (time_errors, .0015)):
                assert errors[0] > errors[1] > errors[2]
                assert errors[-1] < limit
            cells = " & ".join(f"{error:.3e}" for error in length_errors)
            rows.append(f"{d} & {bounces} & {fraction:g} & {cells} " + r"\\")
    Path(__file__).with_name("einstein_geodesic_momentum_rows.tex").write_text(
        "\n".join(rows+[r"\hline"])+"\n", encoding="ascii"
    )


if __name__ == "__main__":
    main()
