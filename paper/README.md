# Geodesic cycles and multiple reflections at spacelike AdS black hole singularities

This working paper calculates the time and length periods of a radial cycle
joining two complex turning points that approach the spacelike singularity of
a planar Schwarzschild–AdS black brane. The numerical check compares the
on-shell actions at the same complex boundary time. A second calculation
obtains the local scalar connection and the transfer for repeated radial
windings. The final calculation solves a tuned scalar problem in a
higher-dimensional Lovelock family, including the two-bounce contribution
to the spectrum and its complex-time poles. Changing the bare scalar mass
at fixed curvature coupling gives a continuous family of exact spectra.

Run the following from this directory:

~~~text
python check_periods.py
python check_connection.py
python check_lovelock.py
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
~~~

The scripts use only the Python standard library. The first checks the
finite-energy periods and the action at fixed boundary time. The second
integrates the scalar modes on rotated rays and checks their transfer
product against Frobenius monodromy. The third integrates the original
nonminimally coupled scalar equation in nine and thirteen dimensions
and compares its retarded spectrum with the exact result, including
four bare-mass variations in nine dimensions.

The Lovelock example gives the boundary response with both endpoint
conditions imposed. The global higher-reflection coefficients for
minimally coupled scalars in the Einstein brane remain open.
