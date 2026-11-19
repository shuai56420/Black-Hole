# Geodesic cycles and multiple reflections at spacelike AdS black hole singularities

This working paper calculates two types of radial geodesic cycles in a
planar Schwarzschild–AdS black brane. The cycle joining two near-singularity
turning points shifts the action at fixed boundary time. The cycle joining
an inner and an outer turning point yields the finite optical round trip.
Repeated inner turns then give limiting boundary times \(nT_*\).
The endpoint Jacobi map gives the transverse response and the Gaussian
factor for contours that combine both cycles.
A second calculation
obtains the local scalar connection and the transfer for repeated radial
windings. The AdS₇ scalar calculation checks the pole shifts and the
second and third harmonics against the full radial equation. The final
calculation solves a tuned scalar problem in a higher-dimensional Lovelock family, including
the two-bounce contribution to the spectrum and its complex-time poles.
Changing the bare scalar mass
at fixed curvature coupling gives a continuous family of exact spectra.
Contour deformation of the full gamma envelope determines the primitive
and mixed-image singularities for arbitrary positive mass parameter.
A residue series gives the jump between lateral continuations of the
mixed thermal images.

Run the following from this directory:

~~~text
python check_periods.py
python check_connection.py
python check_einstein_poles.py
python check_lovelock.py
python check_lovelock_momentum.py
python check_lovelock_crossover.py
python check_lovelock_residues.py
python check_lovelock_scaled_momentum.py
python check_lovelock_momentum_time.py
python check_time_continuation.py
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
~~~

The scripts use only the Python standard library. The first checks the
finite-energy periods, the action at fixed boundary time, the optical
limits of geodesics with two and three inner turns, and their transverse
Jacobi maps. The second
integrates the scalar modes on rotated rays and checks their transfer
product against Frobenius monodromy. The third matches the ingoing and
normalizable scalar series to compute AdS₇ quasinormal frequencies.
It also includes the source series to extract two- and three-reflection
coefficients from the retarded response.
The fourth integrates the original
nonminimally coupled scalar equation in nine and thirteen dimensions
and compares its retarded spectrum with the exact result, including
four bare-mass variations in nine dimensions.
The fifth checks the exact small-momentum dispersion and its Airy limit
against finite-momentum pole matching and high overtones.
The sixth compares the joint mass and overtone limit with the displaced
Airy connection integral and checks the geometric momentum derivative.
The seventh checks the momentum derivative of the retarded response,
its pole residues, and the zero-momentum time-domain mode sum.
The finite scaled-momentum check solves the Airy connection and compares
the shifted pole and two- and three-reflection amplitudes with direct
Frobenius matching at three overtones.
The perturbative time-domain check verifies the reflection harmonics and the
leading coefficient of the pole-motion term at the primitive times.
The final check evaluates the continued Wightman function, checks the leading
singular coefficients for primitive and mixed images, and compares the
residue series with direct integration on tilted contours.

The Lovelock example gives a sheet-resolved continuation of the exact
boundary response. The Einstein calculation checks the repeated
reflections of the high-frequency expansion against the full response
near the quasinormal ray.

The JHEP source archive uses `main.tex` as the master file and includes
the generated tables, bibliography, and bibliography output.
