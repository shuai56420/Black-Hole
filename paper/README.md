# Geodesic cycles and multiple reflections at spacelike AdS black hole singularities

This working paper calculates the time and length periods of a radial cycle
joining two complex turning points that approach the spacelike singularity of
a planar Schwarzschild–AdS black brane. The numerical check compares the
on-shell actions at the same complex boundary time. A second calculation
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
python check_time_continuation.py
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
~~~

The scripts use only the Python standard library. The first checks the
finite-energy periods and the action at fixed boundary time. The second
integrates the scalar modes on rotated rays and checks their transfer
product against Frobenius monodromy. The third matches the ingoing and
normalizable scalar series to compute AdS₇ quasinormal frequencies.
It also includes the source series to extract two- and three-reflection
coefficients from the retarded response.
The fourth integrates the original
nonminimally coupled scalar equation in nine and thirteen dimensions
and compares its retarded spectrum with the exact result, including
four bare-mass variations in nine dimensions.
The fifth evaluates the continued Wightman function, checks the leading
singular coefficients for primitive and mixed images, and compares the
residue series with direct integration on tilted contours.

The Lovelock example gives a sheet-resolved continuation of the exact
boundary response. The Einstein calculation checks the repeated
reflections of the high-frequency expansion against the full response
near the quasinormal ray.

The JHEP source archive uses `main.tex` as the master file and includes
the generated tables, bibliography, and bibliography output.
