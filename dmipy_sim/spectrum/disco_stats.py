"""DiSCo's own radius and density statistics, measured directly from the released substrate (Rafael-Patino,
Girard et al., Data in Brief 38 (2021) 107429, doi:10.1016/j.dib.2021.107429; dataset doi:10.17632/fgf86jdfg6.3,
CC BY 4.0: ``DiSCo_Strands_Trajectories.tck`` and ``DiSCo_Strands_Diameters.txt``), the numbers
:func:`dmipy_sim.spectrum.strands.strands` draws its random strands to match. These are not fitted or guessed:
every value below is a direct measurement of the released files, with the one-line reduction that produced it.

Measured once (``dmipy_sim.io.strands.read_tck`` / ``read_diameters`` on the released files, coordinate unit
25 um, diameter unit 1 mm), 12,196 strands in the released 1 mm^3 cube:

- radius (the released value is the diameter DiSCo calls the INNER one -- the axolemma, :data:`DISCO_G_RATIO`
  scales it to the outer / sheath radius exactly as :func:`dmipy_sim.spec.producers.disco_spec` does): mean
  1.034 um, std 0.363 um, min 0.718 um, max 2.091 um;
- areal number density: 12,196 strands through a 1 mm^2 cross-section (the strands run end to end along the
  cube's long axis, so the count per unit CROSS-SECTIONAL area is what stays constant as the domain's footprint
  grows or shrinks -- 12,196 / mm^2 == 12,196e6 / m^2);
- intra-axonal (inside the released, inner radius) volume fraction of the 1 mm^3 cube: 3.68 %;
- strand length density (sum of every strand's length over the domain volume): 9,750.6 mm / mm^3, i.e. 53.2
  centerline points per mm of strand at the released sampling.

A strand list this dense through a 5 mm cube (25x the cross-section) carries about 25 x 12,196 ~= 305,000
strands, each ~5 mm of centerline at the released point density (~266 points): ~81 M centerline points, the
"~80 M segments" dmipy-sim#697 names as the 5 mm case's memory hazard.
"""

#: m, the released (inner / axolemma) radius statistics of DiSCo's 12,196 strands, measured from
#: ``DiSCo_Strands_Diameters.txt``: ``diameter/2`` over every strand.
DISCO_INNER_RADIUS_STATS_M = dict(mean=1.0341e-6, std=0.3627e-6, min=0.7182e-6, max=2.0913e-6)

#: strands / m^2 of cross-section, DiSCo's own areal density (12,196 strands through the released 1 mm^2
#: cross-section).
DISCO_AREAL_DENSITY_PER_M2 = 12196e6

#: mm of strand centerline per mm^3 of domain, DiSCo's own line density (its 12,196 strands' total length
#: over its 1 mm^3 cube) -- quoted for the generator's docstring and the PR body, not consumed by the code
#: (the areal density plus a strand run end-to-end already fixes it).
DISCO_LINE_DENSITY_PER_MM2 = 9750.6

#: centerline points per mm of strand at DiSCo's own released sampling -- the generator's default step so a
#: generated strand carries the same point density the released tracks do.
DISCO_POINTS_PER_MM = 53.2

#: DiSCo's own number of strands in its released 1 mm^3 cube (Rafael-Patino et al. 2021): the cross-check that
#: :data:`DISCO_AREAL_DENSITY_PER_M2` reproduces at edge = 1 mm.
DISCO_N_STRANDS_PER_MM3 = 12196
