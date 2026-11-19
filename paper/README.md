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

## Build the manuscript

Run from the `paper` directory with pdfLaTeX and BibTeX. The
table inputs are included in the repository.

~~~sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
~~~

This produces `main.pdf`. To create the JHEP source archive from
the same directory, run the appropriate command after the build.

~~~sh
tar -czf jhep-source.tar.gz *.tex *.sty *.bst *.bib main.bbl
~~~

~~~powershell
$sourceFiles = Get-ChildItem -File |
    Where-Object { $_.Extension -in '.tex', '.sty', '.bst', '.bib' } |
    Select-Object -ExpandProperty Name
tar -czf jhep-source.tar.gz @sourceFiles main.bbl
~~~

## Reproduce the numerical tables

Run from the `paper` directory with Python 3, then rebuild the manuscript.

~~~text
python check_periods.py
python check_einstein_geodesic_momentum.py
python check_connection.py
python check_einstein_poles.py
python check_lovelock.py
python check_lovelock_momentum.py
python check_lovelock_crossover.py
python check_lovelock_residues.py
python check_lovelock_scaled_momentum.py
python check_lovelock_geometric_momentum.py
python check_lovelock_momentum_time.py
python check_time_continuation.py
~~~

The scripts use only the Python standard library. The first checks the
finite-energy periods, the action at fixed boundary time, the optical
limits of geodesics with two and three inner turns, and their transverse
Jacobi maps. The finite-separation geodesic calculation matches complex
contours at the same boundary time and prescribed spatial displacement.
It also compares the logarithmic endpoint slopes near the merger of the
inner turning points with direct finite-energy contour integrals.
The connection calculation
integrates the scalar modes on rotated rays and checks their transfer
product against Frobenius monodromy. The AdS₇ calculation matches the ingoing and
normalizable scalar series to compute AdS₇ quasinormal frequencies.
It also includes the source series to extract two- and three-reflection
coefficients from the retarded response.
The Lovelock benchmark integrates the original
nonminimally coupled scalar equation in nine and thirteen dimensions
and compares its retarded spectrum with the exact result, including
four bare-mass variations in nine dimensions.
The small-momentum check tests the exact dispersion and its Airy limit
against finite-momentum pole matching and high overtones.
The crossover check compares the joint mass and overtone limit with the displaced
Airy connection integral and checks the geometric momentum derivative.
The residue check verifies the momentum derivative of the retarded response,
its pole residues, and the zero-momentum time-domain mode sum.
The finite scaled-momentum check solves the Airy connection and compares
the shifted pole and two- and three-reflection amplitudes with direct
Frobenius matching at three overtones.
The geometric momentum check compares the elliptic transverse period
with finite-energy geodesic quadratures and tests the nonlinear optical action
against the shifted Airy connection at large scaled mass.
The perturbative time-domain check verifies the reflection harmonics and the
leading coefficient of the pole-motion term at the primitive times.
The final check evaluates the continued Wightman function at primitive and
mixed images, extracts the leading and first subleading singular terms, and
compares the cut jump from the retarded modes with direct contour integration.

The Lovelock example gives a sheet-resolved continuation of the exact
boundary response. The Einstein calculation checks the repeated
reflections of the high-frequency expansion against the full response
near the quasinormal ray.

The JHEP source archive uses `main.tex` as the master file and includes
the generated tables, bibliography, and bibliography output.
The title page lists three keywords from the
[JHEP keyword list](https://jhep.sissa.it/jhep/help/keywordsList.jsp).
JHEP requires the arXiv identifier of the same manuscript version
and a code availability selection in its submission form.
[JHEP author instructions](https://jhep.sissa.it/jhep/help/JHEP/JHEP_author.jsp)
