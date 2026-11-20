"""Biophysical constants with full citation provenance.

Every default parameter value used anywhere in dmipy-core is catalogued here
with its source publication, DOI, and measurement context.

Schema
------
Each constant has a 'default' dict with the canonical value and its source,
plus an optional 'alternatives' list of values from other experimental
conditions (field strength, species, method).  Every source includes a
'location' field pointing to the specific section/table/figure in the paper
where the value can be found (or 'NEEDS VERIFICATION' if not yet pinpointed).

The membrane water-exchange entries (``kappa_membrane``, ``exchange_time_grey_matter``,
``exchange_rate_grey_matter``, ``exchange_rate_white_matter``, ``intracellular_residence_time``,
``kappa_membrane_direct``) also carry, per source: ``doi``; ``printed`` where the paper prints a
value (that value and its unit as printed; ``value`` is the same number in SI) -- a source whose
value is not printed by its paper (``kappa_membrane``'s default, a 'NEEDS VERIFICATION' source, a
permeability derived by this table) has no ``printed``; ``range`` with ``value`` None where the paper
prints a range; ``temperature_C`` where the paper states it; and ``derived_permeability`` (value,
formula, geometry, radius and who derived it) where a permeability is derived from the measured
quantity.  Their naming rule is stated in the comment above ``kappa_membrane``.
"""

import warnings

from dmipy_analytic.constants import GAMMA

# Shared citation dicts referenced by multiple constants
_CITATION_MILLS1973 = {
    'key': 'mills1973',
    'authors': 'Mills R',
    'title': 'Self-diffusion in normal and heavy water in the range 1-45 deg',
    'journal': 'Journal of Physical Chemistry',
    'year': 1973,
    'doi': '10.1021/j100624a025',
}

_CITATION_MACKAY1994 = {
    'key': 'mackay1994',
    'authors': 'MacKay A, Whittall K, Adler J, Li D, Paty D, Graeb D',
    'title': 'In vivo visualization of myelin water in brain by magnetic resonance',
    'journal': 'Magnetic Resonance in Medicine',
    'year': 1994,
    'doi': '10.1002/mrm.1910310614',
}

_CITATION_ABOITIZ1992 = {
    'key': 'aboitiz1992',
    'authors': 'Aboitiz F, Scheibel AB, Fisher RS, Zaidel E',
    'title': 'Fiber composition of the human corpus callosum',
    'journal': 'Brain Research',
    'year': 1992,
    'doi': '10.1016/0006-8993(92)90178-C',
}

_CITATION_BARAKOVIC2023 = {
    'key': 'barakovic2023',
    'authors': 'Barakovic M, Pizzolato M, Tax CMW, et al.',
    'title': 'Estimating axon radius using diffusion-relaxation MRI: calibrating a surface-based relaxation model with histology',
    'journal': 'Frontiers in Neuroscience',
    'year': 2023,
    'doi': '10.3389/fnins.2023.1209521',
}

_CITATION_JORDANOVA2023 = {
    'key': 'jordanova2023',
    'authors': 'Jordanova KV, Martin MN, Ogier SE, Poorman ME, Keenan KE',
    'title': 'In vivo quantitative MRI: T1 and T2 measurements of the human brain at 0.064 T',
    'journal': 'Magnetic Resonance Materials in Physics, Biology and Medicine (MAGMA)',
    'year': 2023,
    'doi': '10.1007/s10334-023-01095-x',
}

_CITATION_OREILLY2022 = {
    'key': 'oreilly2022',
    'authors': "O'Reilly T, Webb AG",
    'title': 'In vivo T1 and T2 relaxation time maps of brain tissue, skeletal muscle, and lipid measured in healthy '
             'volunteers at 50 mT',
    'journal': 'Magnetic Resonance in Medicine',
    'year': 2022,
    'doi': '10.1002/mrm.29009',
}

_CITATION_COX2010 = {
    'key': 'cox2010',
    'authors': 'Cox EF, Gowland PA',
    'title': "Simultaneous quantification of T2 and T2' using a combined gradient echo-spin echo sequence at ultrahigh field",
    'journal': 'Magnetic Resonance in Medicine',
    'year': 2010,
    'doi': '10.1002/mrm.22522',
}

_CITATION_WEST2018 = {
    'key': 'west2018',
    'authors': 'West KL, Kelm ND, Carson RP, Gochberg DF, Ess KC, Does MD',
    'title': 'Myelin volume fraction imaging with MRI',
    'journal': 'NeuroImage',
    'year': 2018,
    'doi': '10.1016/j.neuroimage.2016.12.067',
}

_CITATION_WIGGERMANN2021 = {
    'key': 'wiggermann2021',
    'authors': 'Wiggermann V, MacKay AL, Rauscher A, Helms G',
    'title': 'In vivo investigation of the multi-exponential T2 decay in human white '
             'matter at 7 T: Implications for myelin water imaging at UHF',
    'journal': 'NMR in Biomedicine',
    'year': 2021,
    'doi': '10.1002/nbm.4429',
}

_CITATION_ROONEY2007 = {
    'key': 'rooney2007',
    'authors': 'Rooney WD, Johnson G, Li X, et al.',
    'title': 'Magnetic field and tissue dependencies of human brain longitudinal 1H2O '
             'relaxation in vivo',
    'journal': 'Magnetic Resonance in Medicine',
    'year': 2007,
    'doi': '10.1002/mrm.21122',
}

_CITATION_RIOUX2015 = {
    'key': 'rioux2015',
    'authors': 'Rioux JA, Levesque IR, Rutt BK',
    'title': 'Biexponential longitudinal relaxation in white matter: characterization '
             'and impact on T1 mapping with IR-FSE and MP2RAGE',
    'journal': 'Magnetic Resonance in Medicine',
    'year': 2015,
    'doi': '10.1002/mrm.25729',
}

_CITATION_KAUPPINEN2025 = {
    'key': 'kauppinen2025',
    'authors': 'Kauppinen RA, et al.',
    'title': 'Fiber orientation-dependent T1 angular features in human white matter at '
             '1.5, 3, and 7 T',
    'journal': 'Magnetic Resonance in Medicine',
    'year': 2025,
    'doi': '10.1002/mrm.70009',
}

_CITATION_SPIJKERMAN2017 = {
    'key': 'spijkerman2017',
    'authors': 'Spijkerman JM, et al.',
    'title': 'T2 mapping of cerebrospinal fluid: 3 T versus 7 T',
    'journal': 'Magnetic Resonance Materials in Physics, Biology and Medicine (MAGMA)',
    'year': 2017,
    'doi': '10.1007/s10334-017-0659-3',
}


_CITATION_WHARTON2012 = {
    'key': 'wharton2012',
    'authors': 'Wharton S, Bowtell R',
    'title': 'Fiber orientation-dependent white matter contrast in gradient echo MRI',
    'journal': 'Proceedings of the National Academy of Sciences',
    'year': 2012,
    'doi': '10.1073/pnas.1211075109',
}


_CITATION_STANISZ2005 = {
    'key': 'stanisz2005',
    'authors': 'Stanisz GJ, Odrobina EE, Pun J, Escaravage M, Graham SJ, Bronskill MJ, Henkelman RM',
    'title': 'T1, T2 relaxation and magnetization transfer in tissue at 3T',
    'journal': 'Magnetic Resonance in Medicine',
    'year': 2005,
    'doi': '10.1002/mrm.20605',
}

_CITATION_TOFTS2003 = {
    'key': 'tofts2003',
    'authors': 'Tofts PS (ed.)',
    'title': 'Quantitative MRI of the Brain: Measuring Changes Caused by Disease',
    'journal': 'John Wiley & Sons (book)',
    'year': 2003,
    'doi': '10.1002/0470869526',
}

BIOPHYSICAL_CONSTANTS = {
    'D_water_37C': {
        'default': {
            'value': 3.05e-9,
            'unit': 'm^2/s',
            'field_T': None,
            'species': 'pure water',
            'method': 'NMR spin-echo self-diffusion',
            'source_key': 'mills1973',
            'location': 'Table I, row T=37 deg C, D(H2O) = 3.05 x 10^-5 cm^2/s',
        },
        'alternatives': [],
        'citation': _CITATION_MILLS1973,
        'description': 'Free water self-diffusion coefficient at 37 deg C (in vivo body temperature)',
    },
    'D_water_25C': {
        'default': {
            'value': 2.299e-9,
            'unit': 'm^2/s',
            'field_T': None,
            'species': 'pure water',
            'method': 'NMR spin-echo self-diffusion',
            'source_key': 'mills1973',
            'location': 'Table I, row T=25 deg C, D(H2O) = 2.299 x 10^-5 cm^2/s',
        },
        'alternatives': [],
        'citation': _CITATION_MILLS1973,
        'description': 'Free water self-diffusion coefficient at 25 deg C (lab temperature)',
    },
    'D_intra_axonal': {
        'default': {
            'value': 1.7e-9,
            'unit': 'm^2/s',
            'field_T': None,
            'species': 'human',
            'method': 'review of DTI studies',
            'source_key': 'beaulieu2002',
            'location': 'Section "Diffusion anisotropy in nerve", '
                        '"the parallel diffusivity ... is approximately 1.7 x 10^-3 mm^2/s"',
        },
        'alternatives': [
            {
                'value': 1.6e-9,
                'unit': 'm^2/s',
                'field_T': 3.0,
                'species': 'human',
                'method': 'NODDI fitting',
                'source_key': 'zhang2012',
                'location': 'Section 2.4, "d_parallel was fixed at 1.7 x 10^-3 mm^2/s"'
                            ' (note: Zhang uses Beaulieu value; fitted values ~1.6 reported in Discussion)',
            },
        ],
        'citation': {
            'key': 'beaulieu2002',
            'authors': 'Beaulieu C',
            'title': 'The basis of anisotropic water diffusion in the nervous system - a technical review',
            'journal': 'NMR in Biomedicine',
            'year': 2002,
            'doi': '10.1002/nbm.782',
        },
        'description': 'In vivo intra-axonal parallel diffusivity',
        'note': 'This is the single-axon (undispersed) parallel diffusivity, NOT the macroscopic '
                'bundle-averaged diffusivity. Macroscopically measured D_parallel from DTI includes '
                'orientation dispersion effects that lower the apparent value. Using a macroscopic '
                'measurement here would double-count dispersion when combined with Watson/Bingham '
                'distributions in models like NODDI. The true single-axon D_par may be slightly '
                'higher than 1.7e-9 m^2/s; literature values measured from dispersed bundles '
                'are biased downward. Commonly fixed in NODDI and similar models; true value '
                'may vary by tract and species.',
    },
    'D_csf': {
        'default': {
            'value': 3.0e-9,
            'unit': 'm^2/s',
            'field_T': 1.5,
            'species': 'human',
            'method': 'DTI of ventricular CSF',
            'source_key': 'pierpaoli1996',
            'location': 'Table 1, ventricles row, mean diffusivity = 3.0 +/- 0.2 x 10^-3 mm^2/s',
        },
        'alternatives': [],
        'citation': {
            'key': 'pierpaoli1996',
            'authors': 'Pierpaoli C, Jezzard P, Basser PJ, Barnett A, Di Chiro G',
            'title': 'Diffusion tensor MR imaging of the human brain',
            'journal': 'Radiology',
            'year': 1996,
            'doi': '10.1148/radiology.201.3.8939209',
        },
        'description': 'CSF diffusivity at body temperature',
        'note': 'Slightly lower than free water due to protein content; commonly rounded to 3.0',
    },
    'D_extra_axonal': {
        'default': {
            'value': 1.2e-9,
            'unit': 'm^2/s',
            'field_T': 3.0,
            'species': 'human',
            'method': 'white matter kurtosis model fit',
            'source_key': 'fieremans2011',
            'location': 'Table 2, De,perp column, WM mean ~1.2 x 10^-3 mm^2/s',
        },
        'alternatives': [],
        'citation': {
            'key': 'fieremans2011',
            'authors': 'Fieremans E, Jensen JH, Helpern JA',
            'title': 'White matter characterization with diffusional kurtosis imaging',
            'journal': 'NeuroImage',
            'year': 2011,
            'doi': '10.1016/j.neuroimage.2011.06.006',
        },
        'description': 'Extra-axonal hindered diffusivity (typical WM)',
    },
    'gamma_proton': {
        'default': {
            'value': GAMMA,                                     # the ecosystem's one gyromagnetic ratio
            'unit': 'rad/s/T',
            'field_T': None,
            'species': None,
            'method': 'CODATA fundamental constants',
            'source_key': 'codata2018',
            'location': 'Table XXXIII, proton gyromagnetic ratio '
                        'gamma_p/2pi = 42.577 478 518 MHz/T; '
                        'gamma_p = 2pi * 42.577 478 518 x 10^6 = 267.522 x 10^6 rad/s/T',
        },
        'alternatives': [],
        'citation': {
            'key': 'codata2018',
            'authors': 'CODATA',
            'title': 'CODATA recommended values of the fundamental physical constants: 2018',
            'journal': 'Reviews of Modern Physics',
            'year': 2021,
            'doi': '10.1103/RevModPhys.93.025010',
        },
        'description': 'Proton gyromagnetic ratio',
    },
    'T2_intra_axonal': {
        'default': {
            'value': 0.070,
            'unit': 's',
            'field_T': 1.5,
            'species': 'human',
            'method': 'multi-component T2 relaxometry',
            'source_key': 'mackay1994',
            'location': 'Figure 3 and Section "Results", long T2 component of WM '
                        '= 70-80 ms (identified as intra/extra-axonal water)',
        },
        'alternatives': [
            {
                'value': 0.050,
                'unit': 's',
                'field_T': 3.0,
                'species': 'human',
                'method': 'diffusion-relaxation MRI with surface relaxation model',
                'source_key': 'barakovic2023',
                'location': 'Table 2, intra-axonal T2 estimates ~50 ms at 3T',
            },
            {
                'value': 0.047,
                'unit': 's',
                'field_T': 7.0,
                'species': 'human',
                'method': 'multi-component GraSE + EPG (spin-echo T2, not T2*)',
                'source_key': 'wiggermann2021',
                'location': 'Fig 5a / Suppl. Table S1: intermediate (intra+extra-cellular, '
                            '"IE") geometric-mean T2 ~47 ms at 7T. Intra- and extra-axonal '
                            'are NOT separately resolved at 7T -- the IE pool value is used '
                            'for both compartments (no measured 7T split).',
            },
        ],
        'citation': _CITATION_MACKAY1994,
        'description': 'T2 relaxation time of intra-axonal water',
        'note': 'T2 is field-dependent; decreases with increasing B0. '
                'MacKay 1.5T; Barakovic 3T (~50 ms); Wiggermann 7T (~47 ms IE pool, '
                'intra/extra not separately resolved -- both set to the IE value).',
    },
    # ---- quantitative MT (two-pool) ----------------------------------------------------------------
    # The two parameters an EMERGENT MT walk needs are kappa_MT (surface reactivity, m/s) and dwell_time
    # (s), neither of which is measured directly. What qMT reports is the macromolecular pool fraction M0B
    # and the fundamental exchange rate R, from which the pseudo-first-order rates follow as
    # k_f = R*M0B and k_r = R*M0A (M0A = 1 - M0B); detailed balance k_f*M0A = k_r*M0B then holds and
    # f_bound = k_f/(k_f + k_r) recovers M0B. Convert with dmipy_analytic.relaxation.kappa_MT_from_forward_rate (which needs the
    # substrate's own S/V, so kappa_MT is geometry-dependent) and dmipy_analytic.relaxation.dwell_time_from_fraction.
    'mt_bound_pool_fraction': {
        'default': {
            'value': 0.139,
            'unit': 'dimensionless',
            'field_T': 3.0,
            'species': 'bovine',
            'method': 'quantitative MT, super-Lorentzian lineshape, in vitro at 37 C',
            'source_key': 'stanisz2005',
            'location': 'Table 2, white matter row: M0B = 13.9 +/- 2.8 %',
        },
        'alternatives': [
            {
                'value': 0.050,
                'unit': 'dimensionless',
                'field_T': 3.0,
                'species': 'bovine',
                'method': 'quantitative MT, in vitro at 37 C (GREY matter)',
                'source_key': 'stanisz2005',
                'location': 'Table 2, gray matter row: M0B = 5.0 +/- 0.5 %',
            },
        ],
        'citation': _CITATION_STANISZ2005,
    },

    'mt_exchange_rate': {
        'default': {
            'value': 23.0,
            'unit': '1/s',
            'field_T': 3.0,
            'species': 'bovine',
            'method': 'quantitative MT, super-Lorentzian lineshape, in vitro at 37 C',
            'source_key': 'stanisz2005',
            'location': 'Table 2, white matter row: R = 23 +/- 4 s^-1. This is the FUNDAMENTAL rate '
                        'constant, not a pseudo-first-order rate: k_f = R*M0B, k_r = R*M0A.',
        },
        'alternatives': [
            {
                'value': 40.0,
                'unit': '1/s',
                'field_T': 3.0,
                'species': 'bovine',
                'method': 'quantitative MT, in vitro at 37 C (GREY matter)',
                'source_key': 'stanisz2005',
                'location': 'Table 2, gray matter row: R = 40 +/- 1 s^-1',
            },
        ],
        'citation': _CITATION_STANISZ2005,
    },

    'T2_bound_pool': {
        'default': {
            'value': 1.0e-5,
            'unit': 's',
            'field_T': 3.0,
            'species': 'bovine',
            'method': 'quantitative MT, super-Lorentzian lineshape, in vitro at 37 C',
            'source_key': 'stanisz2005',
            'location': 'Table 2, white matter row: T2B = 10.0 +/- 1.0 us',
        },
        'alternatives': [
            {
                'value': 9.1e-6,
                'unit': 's',
                'field_T': 3.0,
                'species': 'bovine',
                'method': 'quantitative MT, in vitro at 37 C (GREY matter)',
                'source_key': 'stanisz2005',
                'location': 'Table 2, gray matter row: T2B = 9.1 +/- 0.2 us',
            },
        ],
        'citation': _CITATION_STANISZ2005,
    },

    'T2_myelin': {
        'default': {
            'value': 0.015,
            'unit': 's',
            'field_T': 1.5,
            'species': 'human',
            'method': 'multi-component T2 relaxometry',
            'source_key': 'mackay1994',
            'location': 'Figure 3 and Section "Results", short T2 component of WM '
                        '= 10-20 ms (identified as myelin water); mean ~15 ms',
        },
        'alternatives': [
            {
                'value': 0.010,
                'unit': 's',
                'field_T': 3.0,
                'species': 'human',
                'method': 'diffusion-relaxation MRI with surface relaxation model',
                'source_key': 'barakovic2023',
                'location': 'Section 3.2, myelin water T2 estimates ~10 ms at 3T',
            },
            {
                'value': 0.011,
                'unit': 's',
                'field_T': 7.0,
                'species': 'human',
                'method': 'multi-component GraSE + EPG (spin-echo T2, not T2*)',
                'source_key': 'wiggermann2021',
                'location': 'Short-T2 (myelin water) peak ~11-12 ms at 7T (vs ~15 ms at 3T).',
            },
        ],
        'citation': _CITATION_MACKAY1994,
        'description': 'T2 relaxation time of myelin water',
        'note': 'T2 is strongly field-dependent; use field-matched value. '
                'At 3T myelin water T2 ~10 ms; at 7T ~11 ms (Wiggermann 2021).',
    },
    'T2_extra_axonal': {
        'default': {
            'value': 0.080,
            'unit': 's',
            'field_T': 1.5,
            'species': 'human',
            'method': 'multi-component T2 relaxometry',
            'source_key': 'mackay1994',
            'location': 'Figure 3 and Section "Results", long T2 component of WM '
                        '= 70-80 ms (intra + extra-axonal water not separately resolved)',
        },
        'alternatives': [
            {
                'value': 0.055,
                'unit': 's',
                'field_T': 3.0,
                'species': 'human',
                'method': 'diffusion-relaxation MRI with surface relaxation model',
                'source_key': 'barakovic2023',
                'location': 'Table 2, extra-axonal T2 estimates ~55 ms at 3T',
            },
            {
                'value': 0.047,
                'unit': 's',
                'field_T': 7.0,
                'species': 'human',
                'method': 'multi-component GraSE + EPG (spin-echo T2, not T2*)',
                'source_key': 'wiggermann2021',
                'location': 'Fig 5a / Suppl. Table S1: intermediate (intra+extra-cellular, '
                            '"IE") geometric-mean T2 ~47 ms at 7T. Intra/extra not separately '
                            'resolved at 7T -- IE pool value used for both compartments.',
            },
        ],
        'citation': _CITATION_MACKAY1994,
        'description': 'T2 relaxation time of extra-axonal water',
        'note': 'T2 is field-dependent; use field-matched value. MacKay 1.5T (~80 ms); '
                'Barakovic 3T (~55 ms); Wiggermann 7T (~47 ms IE pool, no measured '
                'intra/extra split -- the apparent T2 baseline shortens with B0 mostly '
                'from microscopic relaxation, NOT the mesoscopic susceptibility the model '
                'adds separately; that mesoscopic term is < 10 percent of the 3->7T shift).',
    },
    'T2_white_matter': {
        'default': {
            'value': 0.069,
            'unit': 's',
            'field_T': 3.0,
            'species': 'human',
            'method': 'in vitro relaxometry at 37 C (white matter as ONE pool, NNLS spectrum mean)',
            'source_key': 'stanisz2005',
            'location': 'Table 1, white matter row: T2 = 69 +/- 3 ms at 3 T',
        },
        'alternatives': [
            {
                'value': 0.072,
                'unit': 's',
                'field_T': 1.5,
                'species': 'human',
                'method': 'in vitro relaxometry at 37 C (white matter, single pool)',
                'source_key': 'stanisz2005',
                'location': 'Table 1, white matter row: T2 = 72 +/- 4 ms at 1.5 T',
            },
            {
                'value': 0.081,
                'unit': 's',
                'field_T': 0.064,
                'species': 'human',
                'method': 'in vivo multi-echo spin echo, 10 volunteers, manual ROI (white matter), Hyperfine Swoop',
                'source_key': 'jordanova2023',
                'location': 'Results and Supplementary Table S.2, All row: "For T2, the manual ROI averages are 0.081 s '
                            '+/- 0.0024 s for WM"; the automatic segmentation could not separate WM from GM in T2 '
                            '(0.097 +/- 0.0016 s combined)',
            },
        ],
        'citation': _CITATION_STANISZ2005,
        'description': 'T2 relaxation time of white matter as ONE pool -- a whole-tissue figure, which is '
                       'what a segmented image reports and NOT the three-pool decomposition. A replay of a '
                       'myelinated substrate uses T2_intra_axonal / T2_extra_axonal / T2_myelin instead; '
                       'this constant is for a white-matter voxel modelled without that structure.',
    },
    'T2_grey_matter': {
        'default': {
            'value': 0.099,
            'unit': 's',
            'field_T': 3.0,
            'species': 'human',
            'method': 'in vitro relaxometry of fresh bovine/human tissue at 37 C (grey matter, single pool)',
            'source_key': 'stanisz2005',
            'location': 'Table 1, grey matter row: T2 = 99 +/- 7 ms at 3 T',
        },
        'alternatives': [
            {
                'value': 0.095,
                'unit': 's',
                'field_T': 1.5,
                'species': 'human',
                'method': 'in vitro relaxometry at 37 C (grey matter, single pool)',
                'source_key': 'stanisz2005',
                'location': 'Table 1, grey matter row: T2 = 95 +/- 8 ms at 1.5 T',
            },
            {
                'value': 0.105,
                'unit': 's',
                'field_T': 0.064,
                'species': 'human',
                'method': 'in vivo multi-echo spin echo, 10 volunteers, manual ROI (grey matter), Hyperfine Swoop',
                'source_key': 'jordanova2023',
                'location': 'Results and Supplementary Table S.2, All row: "0.105 s +/- 0.0241 s for GM" (manual ROI); '
                            'the automatic segmentation could not separate WM from GM in T2. The spread across '
                            'volunteers is a quarter of the value.',
            },
            {
                'value': 0.047,
                'unit': 's',
                'field_T': 7.0,
                'species': 'human',
                'method': 'in vivo combined gradient echo-spin echo at 7 T (frontal grey matter)',
                'source_key': 'cox2010',
                'location': 'Abstract: "T2 having a linear dependence (frontal gray matter: 87, 76, 47 ms, occipital '
                            'gray matter: 80, 68, 46 ms and white matter: 80, 71, 47 ms at 1.5, 3.0 and 7.0 T, '
                            'respectively)"',
                'note': 'The abstract values; the table with the uncertainties was not accessible when this was '
                        'recorded (Wiley full text).',
            },
        ],
        'citation': _CITATION_STANISZ2005,
        'description': 'T2 relaxation time of grey matter as one pool (a sphere-packed cortex substrate '
                       'replays both of its pools at it)',
    },
    'T1_white_matter': {
        'default': {
            'value': 1.084,
            'unit': 's',
            'field_T': 3.0,
            'species': 'human',
            'method': 'in vitro inversion recovery at 37 C (white matter as ONE pool, mono-exponential)',
            'source_key': 'stanisz2005',
            'location': 'Table 1, white matter row: T1 = 1084 +/- 45 ms at 3 T',
        },
        'alternatives': [
            {
                'value': 0.884,
                'unit': 's',
                'field_T': 1.5,
                'species': 'human',
                'method': 'in vitro inversion recovery at 37 C (white matter, single pool)',
                'source_key': 'stanisz2005',
                'location': 'Table 1, white matter row: T1 = 884 +/- 50 ms at 1.5 T',
            },
            {
                'value': 0.294,
                'unit': 's',
                'field_T': 0.064,
                'species': 'human',
                'method': 'in vivo inversion recovery, 10 volunteers, automatic segmentation (white matter), Hyperfine Swoop',
                'source_key': 'jordanova2023',
                'location': 'Results and Supplementary Table S.2, All row: "the automatically segmented averages are '
                            '0.294 s +/- 0.0179 s for WM"; the manual-ROI average is 0.254 +/- 0.0179 s',
                'note': 'The 0.275 s sometimes quoted beside it is O\'Reilly and Webb\'s 50 mT value, which Jordanova '
                        'quotes in the Discussion; it is not a 0.064 T measurement.',
            },
        ],
        'citation': _CITATION_STANISZ2005,
        'description': 'T1 relaxation time of white matter as ONE pool -- the whole-tissue figure a segmented '
                       'image reports, NOT the three-pool decomposition (T1_intra_axonal / T1_extra_axonal / '
                       'T1_myelin). T1 falls steeply below 1 T, so a field-matched read matters more here '
                       'than it does for T2.',
    },
    'T1_grey_matter': {
        'default': {
            'value': 1.820,
            'unit': 's',
            'field_T': 3.0,
            'species': 'human',
            'method': 'in vitro relaxometry at 37 C (grey matter, single pool)',
            'source_key': 'stanisz2005',
            'location': 'Table 1, grey matter row: T1 = 1820 +/- 114 ms at 3 T',
        },
        'alternatives': [
            {
                'value': 0.460,
                'unit': 's',
                'field_T': 0.064,
                'species': 'human',
                'method': 'in vivo inversion recovery, 10 volunteers, automatic segmentation (grey matter), Hyperfine Swoop',
                'source_key': 'jordanova2023',
                'location': 'Results and Supplementary Table S.2, All row: "0.46 s +/- 0.1258 s for GM" (automatic '
                            'segmentation); the manual-ROI average is 0.377 +/- 0.0351 s',
                'note': 'The 0.327 s sometimes quoted beside it is O\'Reilly and Webb\'s 50 mT value, which Jordanova '
                        'quotes in the Discussion; it is not a 0.064 T measurement.',
            },
            {
                'value': 2.132,
                'unit': 's',
                'field_T': 7.0,
                'species': 'human',
                'method': 'Look-Locker adiabatic inversion recovery at 7 T (cortical grey matter)',
                'source_key': 'rooney2007',
                'location': 'Table 1, row 7.0 T: cortical GM T1 = 2132 +/- 103 ms (WM 1220 +/- 36, caudate 1745 +/- 64, '
                            'putamen 1700 +/- 66, thalamus 1656 +/- 84 ms)',
            },
            {
                'value': 1.124,
                'unit': 's',
                'field_T': 1.5,
                'species': 'human',
                'method': 'in vitro relaxometry at 37 C (grey matter, single pool)',
                'source_key': 'stanisz2005',
                'location': 'Table 1, grey matter row: T1 = 1124 +/- 50 ms at 1.5 T',
            },
        ],
        'citation': _CITATION_STANISZ2005,
        'description': 'T1 relaxation time of grey matter as one pool',
    },
    'proton_density_white_matter': {
        'default': {
            'value': 0.70,
            'unit': '(water content relative to CSF = 1)',
            'field_T': None,
            'species': 'human',
            'method': 'quantitative proton-density / water-content imaging',
            'source_key': 'tofts2003',
            'location': 'Chapter 4 (proton density and water content), tissue water content: white matter '
                        '~0.70 of CSF -- NEEDS VERIFICATION of the table number',
        },
        'alternatives': [],
        'citation': _CITATION_TOFTS2003,
        'description': 'Proton density (water content) of white matter relative to CSF: the m0 a white-matter '
                       'substrate declares in a replay phantom',
    },
    'proton_density_grey_matter': {
        'default': {
            'value': 0.85,
            'unit': '(water content relative to CSF = 1)',
            'field_T': None,
            'species': 'human',
            'method': 'quantitative proton-density / water-content imaging',
            'source_key': 'tofts2003',
            'location': 'Chapter 4 (proton density and water content), tissue water content: grey matter '
                        '~0.85 of CSF -- NEEDS VERIFICATION of the table number',
        },
        'alternatives': [],
        'citation': _CITATION_TOFTS2003,
        'description': 'Proton density (water content) of grey matter relative to CSF: the m0 a grey-matter '
                       'substrate declares in a replay phantom',
    },
    'proton_density_csf': {
        'default': {
            'value': 1.0,
            'unit': '(water content relative to CSF = 1)',
            'field_T': None,
            'species': 'human',
            'method': 'definition: CSF is the reference water content',
            'source_key': 'tofts2003',
            'location': 'Chapter 4: CSF is the 100 % water reference the tissue values are given against',
        },
        'alternatives': [],
        'citation': _CITATION_TOFTS2003,
        'description': 'Proton density of CSF, the reference (1.0): the m0 free water declares in a replay phantom',
    },
    'T2_csf': {
        'default': {
            'value': 2.0,
            'unit': 's',
            'field_T': 3.0,
            'species': 'human',
            'method': 'T2 mapping of ventricular CSF',
            'source_key': 'piechnik2009',
            'location': 'Section "Results", ventricular CSF T2 ~2000 ms at 3T',
        },
        'alternatives': [
            {
                'value': 1.0,
                'unit': 's',
                'field_T': 7.0,
                'species': 'human',
                'method': 'T2-prep CSF mapping',
                'source_key': 'spijkerman2017',
                'location': 'Ventricular CSF T2 ~1.0 s at 7T -- HALVES from ~2.0 s at 3T '
                            '(Spijkerman 2017). The 3T fallback would be ~2x too long here.',
            },
            {
                'value': 1.172,
                'unit': 's',
                'field_T': 0.064,
                'species': 'human',
                'method': 'in vivo multi-echo spin echo, 10 volunteers, manual ROI (CSF), Hyperfine Swoop',
                'source_key': 'jordanova2023',
                'location': 'Results and Supplementary Table S.2, All row: "1.172 s +/- 0.3234 s for CSF" (manual ROI); '
                            'the automatic segmentation gives 0.553 +/- 0.1422 s, and the authors state that their '
                            'protocol underestimates the T2 of CSF. The spread is a quarter of the value.',
            },
        ],
        'citation': {
            'key': 'piechnik2009',
            'authors': 'Piechnik SK, Evans J, Bary LH, Wise RG, Jezzard P',
            'title': 'Functional changes in CSF volume estimated using measurement of water T2 relaxation',
            'journal': 'Magnetic Resonance in Medicine',
            'year': 2009,
            'doi': '10.1002/mrm.21897',
        },
        'description': 'T2 relaxation time of CSF',
    },
    'T1_intra_axonal': {
        'default': {
            'value': 1.2,
            'unit': 's',
            'field_T': 3.0,
            'species': 'human',
            'method': 'IR-EPI T1 mapping',
            'source_key': 'wright2008',
            'location': 'Table 2, WM row at 3T, T1 = 1084 +/- 45 ms (IR-EPI)',
        },
        'alternatives': [
            {
                'value': 1.35,
                'unit': 's',
                'field_T': 7.0,
                'species': 'human',
                'method': 'biexponential IR-FSE, long (aqueous/IE) T1 component',
                'source_key': 'rioux2015',
                'location': 'Long/aqueous (intra+extra-cellular, "IE") T1 = 1349 +/- 18 ms '
                            'at 7T. Excludes the fast myelin pool (modelled separately), so it '
                            'is the per-compartment IE target. Intra/extra not separately '
                            'resolved -- IE value used for both. Within-study cross-check: '
                            'Wright 2008 (this entry\'s 3T source) also reports WM T1 at 7T.',
            },
        ],
        'citation': {
            'key': 'wright2008',
            'authors': 'Wright PJ, Mougin OE, Totman JJ, et al.',
            'title': 'Water proton T1 measurements in brain tissue at 7, 3, and 1.5 T '
                     'using IR-EPI, IR-TSE, and MPRAGE: results and optimization',
            'journal': 'Magnetic Resonance Materials in Physics, Biology and Medicine (MAGMA)',
            'year': 2008,
            'doi': '10.1007/s10334-008-0104-8',  # CrossRef-verified 2026-06-10 (was 10.1002/mrm.21513 = Setsompop 2008)
        },
        'description': 'T1 relaxation time of white matter at 3T',
        'note': 'Used as proxy for intra-axonal T1; true compartment-specific T1 '
                'requires multi-compartment T1 mapping',
    },
    'T1_myelin': {
        'default': {
            'value': 0.440,
            'unit': 's',
            'field_T': 3.0,
            'species': 'human',
            'method': 'mcDESPOT multi-component T1-T2 estimation, in vivo 3T Siemens Tim Trio',
            'source_key': 'deoni2014',
            'location': 'Discussion: "Values obtained herein are 440 +/- 21 ms using the '
                        'default boundaries and 442 +/- 20 ms using the expanded T1,M boundaries." '
                        'In vivo fitted T1,M (myelin water component) with search bounds [300, 650] ms.',
        },
        'alternatives': [
            {
                'value': 0.387,
                'unit': 's',
                'field_T': 1.5,
                'species': 'human',
                'method': 'mcDESPOT, in vivo 1.5T',
                'source_key': 'deoni2010',
                'location': 'Results/Table: frontal WM T1,M = 387 +/- 26 ms at 1.5T',
                'citation': {
                    'key': 'deoni2010',
                    'authors': 'Deoni SCL',
                    'title': 'Correction of main and transmit magnetic field (B0 and B1) '
                             'inhomogeneity effects in multicomponent-driven equilibrium '
                             'single-pulse observation of T1 and T2',
                    'journal': 'Magnetic Resonance in Medicine',
                    'year': 2011,
                    'doi': '10.1002/mrm.22578',
                },
            },
            {
                'value': 0.55,
                'unit': 's',
                'field_T': 7.0,
                'species': 'human',
                'method': 'MP2RAGE T1 relaxogram, short (myelin-associated) component',
                'source_key': 'kauppinen2025',
                'location': 'Short (myelin-associated) T1 relaxogram component ~540 ms at 7T '
                            '(120/260/540 ms at 1.5/3/7T). CONTESTED/method-dependent: the '
                            'IR-FSE short component is ~57 ms (Rioux 2015) -- a different pool. '
                            'Low-medium confidence; an alternative is keeping ~0.44 s.',
            },
        ],
        'citation': {
            'key': 'deoni2014',
            'authors': 'Deoni SCL, Kolind SH',
            'title': 'Investigating the stability of mcDESPOT myelin water fraction values '
                     'derived using a stochastic region contraction approach',
            'journal': 'Magnetic Resonance in Medicine',
            'year': 2015,
            'doi': '10.1002/mrm.25108',  # CrossRef-verified 2026-06-10 (was 10.1002/mrm.25141 = Corum 2015)
        },
        'description': 'T1 relaxation time of myelin water at 3T, in vivo human WM',
        'note': 'T1 is field-dependent; myelin water T1 is substantially shorter than '
                'intra/extra-axonal T1. The former "lancashire2015" attribution was a '
                'phantom reference and has been corrected to Deoni & Kolind 2015 MRM. '
                'The Barakovic 2023 alternative ("~300 ms") was also incorrect — that '
                'paper reports intra-axonal T1a (650–760 ms), not myelin water T1.',
    },
    'T1_extra_axonal': {
        'default': {
            'value': 1.0,
            'unit': 's',
            'field_T': 3.0,
            'species': 'human',
            'method': 'mcDESPOT simulation ground truth (combined intra+extra-axonal IE compartment)',
            'source_key': 'deoni2012',
            'location': 'Methods "Simulation Methods": T1,IE ground truth = 965 ms. '
                        'Results "Simulation Results": fitted T1,IE = 979 +/- 18 ms. '
                        'Note: literature does not separately resolve T1_intra vs T1_extra; '
                        'both are treated as a single IE compartment.',
        },
        'alternatives': [
            {
                'value': 1.084,
                'unit': 's',
                'field_T': 3.0,
                'species': 'human',
                'method': 'IR-EPI T1 mapping, whole WM (not compartment-resolved)',
                'source_key': 'wright2008',
                'location': 'Table 2, WM at 3T, T1 = 1084 +/- 45 ms — used as proxy for '
                             'intra/extra-axonal T1 in the absence of compartment-specific data',
            },
            {
                'value': 1.35,
                'unit': 's',
                'field_T': 7.0,
                'species': 'human',
                'method': 'biexponential IR-FSE, long (aqueous/IE) T1 component',
                'source_key': 'rioux2015',
                'location': 'Long/aqueous (IE) T1 = 1349 +/- 18 ms at 7T (Rioux 2015); same '
                            'combined IE pool as intra -- no measured intra/extra split at 7T.',
            },
        ],
        'citation': {
            'key': 'deoni2012',
            'authors': 'Deoni SCL, Matthews L, Kolind SH',
            'title': 'One component? Two components? Three? The effect of including a '
                     'nonexchanging "free" water component in multicomponent driven equilibrium '
                     'single pulse observation of T1 and T2',
            'journal': 'Magnetic Resonance in Medicine',
            'year': 2013,
            'doi': '10.1002/mrm.24429',  # CrossRef-verified 2026-06-10 (was 10.1002/mrm.24619 = Beeman 2013)
            'pmcid': 'PMC3711852',
        },
        'description': 'T1 relaxation time of extra-axonal (hindered) water in WM at 3T. '
                       'No direct compartment-specific measurement exists; value from '
                       'mcDESPOT IE-compartment simulation ground truth (Deoni 2012).',
        'note': 'The literature treats intra- and extra-axonal water as a single long-T2 '
                'compartment (IE). A value of ~1.0 s is consistent with the Deoni 2012 '
                'simulation ground truth (965 ms) and the WM T1 map of Wright 2008 '
                '(1084 ms). Use 1.0 s as the default; the small difference from T1_intra '
                '(1.4 s corrected for surface relaxivity) reflects that T1_extra has no '
                'surface relaxivity correction.',
    },
    'T1_csf': {
        'default': {
            'value': 4.0,
            'unit': 's',
            'field_T': 3.0,
            'species': 'human',
            'method': 'T1 mapping',
            'source_key': 'rooney2007',
            'location': 'Table 1, CSF row at 3T, T1 = 4300 +/- 200 ms',
        },
        'alternatives': [
            {
                'value': 4.3,
                'unit': 's',
                'field_T': 7.0,
                'species': 'human',
                'method': 'Look-Locker adiabatic-IR (same subjects, 0.2-7T)',
                'source_key': 'rooney2007',
                'location': 'CSF T1 is field-INDEPENDENT at 4.3 +/- 0.2 s across 0.2-7T '
                            '(Rooney 2007) -- the one relaxation value that does not change '
                            'with B0; the 7T entry equals the 3T one by physics, not fallback.',
            },
            {
                'value': 3.695,
                'unit': 's',
                'field_T': 0.05,
                'species': 'human',
                'method': 'in vivo inversion-recovery turbo spin echo at 50 mT, 3 volunteers (ventricular CSF)',
                'source_key': 'oreilly2022',
                'location': 'Results: "The mean T1 in the CSF is measured to be 3695 +/- 287 ms, and the mean T2 is '
                            '1584 +/- 124 ms"; Table 1 per subject 3528 +/- 192, 4033 +/- 325, 3546 +/- 134 ms',
                'note': 'The nearest measured CSF T1 to 0.064 T: Jordanova 2023 reports 2.07 s (manual) / 1.85 s '
                        '(automatic) at 0.064 T and states that their protocol cannot measure it ("Future work is '
                        'needed to accurately measure T1 relaxation in CSF at 0.064 T").',
            },
        ],
        'citation': {
            'key': 'rooney2007',
            'authors': 'Rooney WD, Johnson G, Li X, et al.',
            'title': 'Magnetic field and tissue dependencies of human brain longitudinal '
                     '1H2O relaxation in vivo',
            'journal': 'Magnetic Resonance in Medicine',
            'year': 2007,
            'doi': '10.1002/mrm.21122',
        },
        'description': 'T1 relaxation time of CSF',
    },
    'D_myelin_radial': {
        'default': {
            'value': 0.2e-9,
            'unit': 'm^2/s',
            'field_T': None,
            'species': 'human',
            'method': 'restricted diffusion across myelin lamellae',
            'source_key': 'mackay1994',
            'location': 'Myelin water is strongly restricted radially across the '
                        'lipid bilayers; radial diffusivity ~0.2 µm²/ms.',
        },
        'alternatives': [],
        'citation': _CITATION_MACKAY1994,
        'description': 'Radial (across-lamellae) diffusivity of myelin water in the '
                       'sheath compartment of the canonical substrate.',
        'note': 'Myelin water has very short T2 (~10 ms at 3T) and is near-invisible '
                'at clinical TE, but still carries volume; included for completeness '
                'in the MC substrate.',
    },
    'D_myelin_tangential': {
        'default': {
            'value': 1.0e-9,
            'unit': 'm^2/s',
            'field_T': None,
            'species': 'human',
            'method': 'tangential diffusion along myelin lamellae',
            'source_key': 'mackay1994',
            'location': 'Tangential (along-lamellae) myelin-water diffusivity '
                        '~1.0 µm²/ms, larger than radial.',
        },
        'alternatives': [],
        'citation': _CITATION_MACKAY1994,
        'description': 'Tangential (along-lamellae) diffusivity of myelin water in '
                       'the sheath compartment of the canonical substrate.',
        'note': 'Pairs with D_myelin_radial; the myelin sheath is an anisotropic '
                'annular compartment.',
    },
    'chi_iso_myelin': {
        'default': {
            'value': -0.1e-6,
            'unit': 'SI (dimensionless)',
            'field_T': None,   # susceptibility is field-independent
            'species': 'human',
            'method': 'GRE phase + hollow-cylinder model',
            'source_key': 'wharton2012',
            'location': 'Table 2, row chi_I + E + chi_A: chi_I = -100 +/- 20 ppb (Fig. 3C defines chi_I as the isotropic part of '
                        'the tensor); Table 3 refits give chi_I = -60 +/- 20 ppb',
        },
        'alternatives': [],
        'citation': _CITATION_WHARTON2012,
        'description': 'Isotropic (orientation-independent) volume susceptibility of myelin relative to water: '
                       'the chi_iso replay knob of the field tier. The Winther meshes were built and published '
                       'with chi_myelin - chi_water = +1.06e-6 (their own convention), which winther_spec keeps.',
        'note': 'Pairs with delta_chi_a_myelin (chi_A). The isotropic part is the same in both conventions, so '
                'the value is Wharton & Bowtell 2012\'s as printed (Table 2, row chi_I + E + chi_A).',
    },
    'delta_chi_a_myelin': {
        'default': {
            # chi_A in THIS package's tensor, chi = chi_I I + chi_A (n n^T - I/3), chi_A = chi_radial - chi_tangential.
            # Wharton & Bowtell's tensor (their Fig. 3C) is chi_I I + diag(chi_A_WB, -chi_A_WB/2, -chi_A_WB/2) in the
            # radial frame, whose radial - tangential difference is (3/2) chi_A_WB, so chi_A = (3/2) chi_A_WB.
            'value': -0.15e-6,
            'unit': 'SI (dimensionless)',
            'field_T': None,   # susceptibility is field-independent
            'species': 'human',
            'method': 'GRE phase + hollow-cylinder model, converted to the radial - tangential convention',
            'source_key': 'wharton2012',
            'location': 'Table 2, row chi_I + E + chi_A: chi_A_WB = -100 +/- 20 ppb (tensor defined in Fig. 3C); Table 3 refits '
                        'give chi_A_WB = -120 +/- 20 ppb. NOT Table 1, which lists the frequency-perturbation expressions',
            'reported_value': -0.1e-6,
            'reported_convention': 'diag(chi_A, -chi_A/2, -chi_A/2) in the radial frame (Wharton & Bowtell Fig. 3C)',
            'conversion': 'chi_A (radial - tangential) = (3/2) chi_A_WB',
        },
        'alternatives': [
            {
                'value': -0.08e-6,
                'unit': 'SI (dimensionless)',
                'field_T': 7.0,
                'species': 'human',
                'method': 'Multi-echo GRE phase at 7T + susceptibility tensor imaging',
                'source_key': 'he2009',
                'location': 'Fig. 3, WM susceptibility anisotropy estimate',
                'citation': {
                    'key': 'he2009',
                    'authors': 'He X, Yablonskiy DA',
                    'title': 'Biophysical mechanisms of phase contrast in gradient echo MRI',
                    'journal': 'Proceedings of the National Academy of Sciences',
                    'year': 2009,
                    'doi': '10.1073/pnas.0903111106',
                },
            },
            {
                'value': -0.1e-6,
                'unit': 'SI (dimensionless)',
                'field_T': 7.0,
                'species': 'human',
                'method': 'Susceptibility tensor imaging (STI) at 7T',
                'source_key': 'liu2010_sti',
                'location': 'Fig. 5, principal susceptibility values in WM tracts; '
                            'Δχ = χ_∥ − χ_⊥ ≈ −0.1 ppm',
                'citation': {
                    'key': 'liu2010_sti',
                    'authors': 'Liu C',
                    'title': 'Susceptibility tensor imaging',
                    'journal': 'Magnetic Resonance in Medicine',
                    'year': 2010,
                    'doi': '10.1002/mrm.22391',
                },
            },
        ],
        'citation': _CITATION_WHARTON2012,
        'description': (
            'Susceptibility anisotropy of myelinated white matter: '
            'Δχ_a = χ_radial − χ_tangential of the sheath tensor χ = χ_I·I + Δχ_a(n nᵀ − I/3), n the radial '
            'unit vector (Wharton & Bowtell\'s χ_A is two thirds of it).  Negative because the lipid chains '
            'are radially aligned (phospholipid bilayer geometry).  '
            'In SI units (dimensionless); 1 ppm = 1×10⁻⁶.  '
            'Enters the hollow-cylinder dipolar field as: '
            'ΔB_ea = (Δχ_a·B₀·sin²θ/2)·b²(1−g²)/r²·cos(2φ).'
        ),
        'note': (
            'Sign convention: negative Δχ_a means fibres parallel to B₀ are '
            'slightly less diamagnetic than fibres perpendicular to B₀.  '
            'Wharton & Bowtell report χ_A = −0.1 ppm in their tensor, which is −0.15 ppm here; the '
            'alternatives below are other studies\' estimates, quoted as printed and not converted.  '
            'Value is field-strength independent (susceptibility is a static '
            'tissue property); the field strength B₀ appears explicitly in the '
            'field formula, not here.'
        ),
    },
    'rho1_axon_membrane': {
        'default': {
            'value': 8.7e-8,
            'unit': 'm/s',
            'field_T': None,   # surface relaxivity is a membrane material property, treated field-independent (calibrated ex vivo)
            'species': 'human',
            'method': 'diffusion-relaxation MRI calibrated against histology (ex vivo corpus callosum)',
            'source_key': 'barakovic2023',
            'location': 'Brownstein-Tarr T1 surface relaxivity fit; 1/T1a = 1/T1c + 2*rho1/r. '
                        'In vivo 3T correction: T1_apparent=1.2s (Wright 2008), R=1.46um gives '
                        'T1_bulk_intra = 1/(1/1.2 - 2*rho1/R) ~ 1.40 s. '
                        'Practically relevant for PGSTE sequences (mixing time TM); '
                        'negligible for PGSE where T1 >> Delta.',
        },
        'alternatives': [],
        'citation': {
            'key': 'barakovic2023',
            'authors': 'Barakovic M, Pizzolato M, Tax CMW, Rudrapatna U, Magon S, '
                       'Dyrby TB, Granziera C, Thiran JP, Jones DK, Canales-Rodriguez EJ',
            'title': 'Estimating axon radius using diffusion-relaxation MRI: calibrating '
                     'a surface-based relaxation model with histology',
            'journal': 'Frontiers in Neuroscience',
            'year': 2023,
            'doi': '10.3389/fnins.2023.1209521',
        },
        'description': 'T1 surface relaxivity of the axon membrane (axolemma). '
                       'Use paired with corrected T1_bulk_intra (~1.40 s), not T1_apparent.',
    },
    # ── Membrane water exchange ──────────────────────────────────────────────────────────
    # Naming rule: an entry is named by the quantity the paper MEASURED, followed by the tissue
    # class when the quantity is a tissue average (exchange_time_grey_matter,
    # exchange_rate_grey_matter, exchange_rate_white_matter); a per-cell quantity carries the
    # cell type in a 'cell_type' field instead (intracellular_residence_time), and a directly
    # measured membrane permeability is kappa_membrane_direct.  An exchange time, an exchange
    # rate (AXR) and a residence time are NOT permeabilities.  A permeability derived from one
    # is stored beside it under 'derived_permeability' with the formula, the geometry, the
    # radius and who derived it.  For a well-mixed interior the intracellular residence time is
    #     cylinder of radius R:  tau_i = R / (2 kappa)      sphere of radius R:  tau_i = R / (3 kappa)
    # (Nedjati-Gilani 2017 Section 2.2 writes the cylinder form as k = R / (2 tau_i); Yang 2018
    # Eq. [1] writes the general form P_d = V_i / (A tau_i)).  The two exchange quantities differ
    # from tau_i: NEXI's t_ex = 1 / (r_ie + r_ei) (Jelescu 2022 Section 2.1.2), which is
    # (1 - f) tau_i for an intra-neurite fraction f, and FEXI's AXR is the apparent estimate of
    # k = k_sf + k_fs (Lampinen 2017, Theory, Eqs. (1)-(3)), the SUM of the forward and reverse
    # rates between the slow and the fast diffusion components, which need not be the intra- and
    # extracellular spaces.
    # 'printed' (present where the paper prints the value) is the value with its unit as the paper
    # prints it; 'value' is that number in SI (ms x 1e-3 = s; cm/s x 1e-2 = m/s; mm/s x 1e-3 = m/s;
    # um/ms x 1e-3 = m/s; um/s x 1e-6 = m/s); a printed range is 'range' with 'value' None.
    'kappa_membrane': {
        'default': {
            'value': 1e-5,
            'unit': 'm/s',
            'field_T': None,   # membrane permeability is geometric, field-independent
            'species': None,   # not applicable: no measurement produced this value
            'method': 'none: 1e-5 m/s is not printed by the cited paper (see note)',
            'source_key': 'nilsson2013',
            'doi': '10.1002/mrm.24395',
            'value_in_cited_paper': False,
            'location': 'ABSTRACT ONLY (the full text was not reached; see note). The abstract prints: '
                        '"Apparent exchange rates in white matter range from 0.8±0.08 s(-1) in the internal '
                        'capsule, to 1.6±0.11 s(-1) for frontal white matter, indicating that low values are '
                        'associated with high myelination." and "In white matter, the apparent exchange rate '
                        'values suggest intra-axonal exchange times in the order of seconds". No permeability '
                        'and no radius appear in the abstract.',
        },
        'alternatives': [],
        'citation': {
            'key': 'nilsson2013',
            'authors': 'Nilsson M, Latt J, van Westen D, Brockstedt S, Lasic S, '
                       'Stahlberg F, Topgaard D',
            'title': 'Noninvasive mapping of water diffusional exchange in the human '
                     'brain using filter-exchange imaging',
            'journal': 'Magnetic Resonance in Medicine',
            'year': 2013,
            'doi': '10.1002/mrm.24395',
        },
        'description': 'Axonal membrane permeability used by canonical_white_matter() (the walk\'s '
                       'kappa_inner of Substrate.canonical()).',
        'note': 'The value 1e-5 m/s is NOT given by Nilsson et al. 2013: the paper reports apparent exchange '
                'rates (AXR) of 0.8 and 1.6 s^-1 in white matter, not ~10 s^-1, and no permeability. The '
                'conversion the entry formerly stated, kappa = k R / 3, would need R = 3 kappa / k = '
                '3 x 1e-5 / 0.8 = 37.5 um (internal capsule) or 3 x 1e-5 / 1.6 = 18.75 um (frontal WM) to '
                'produce 1e-5 m/s; the cylinder form kappa = k R / 2 would need 25 um or 12.5 um. Both are '
                'tens of times the radius of a white-matter axon, and an AXR is in any case the sum of two '
                'rates between diffusion components, not an intra-axonal exit rate (see the membrane-exchange '
                'block comment). The value is kept as the default only so that canonical_white_matter() and '
                'Substrate.canonical() do not move silently; the measured quantities are in '
                'exchange_rate_white_matter, intracellular_residence_time and kappa_membrane_direct. '
                'The full text of Nilsson 2013 was not reached, so the location quotes the abstract.',
    },
    'exchange_time_grey_matter': {
        'default': {
            'value': 42.3e-3,
            'unit': 's',
            'printed': '42.3 [40.0, 44.7] ms (mean and 95% CI of the per-ROI medians)',
            'field_T': None,
            'species': 'human',
            'temperature_C': None,   # in vivo; the paper states no temperature
            'method': 'NEXI_RM (NEXI with Rician-mean correction), in vivo, 300 mT/m Connectom 3 T, '
                      'b up to 7.5 ms/um^2, Delta 20-49 ms, 4 healthy adults',
            'region': 'cortical grey matter, DKT atlas ROIs',
            'source_key': 'uhl2024',
            'doi': '10.1162/imag_a_00104',
            'location': 'Section 3.2 "Experimental", Table 3 ("Mean estimates and 95% confidence intervals of '
                        'the median in every ROI of the DKT atlas using NEXI, NEXI dot , corrected for rice '
                        'mean (RM) or not."), row "NEXI RM", column "t ex (ms)": "42.3 [40.0, 44.7]". The '
                        'model with the lowest AICc in that table (-143.0 ± 16.0).',
        },
        'alternatives': [
            {'value': 103.9e-3, 'unit': 's', 'printed': '103.9 [100.3, 107.5] ms', 'field_T': None,
             'species': 'human', 'method': 'NEXI (no Rician correction), in vivo', 'source_key': 'uhl2024',
             'doi': '10.1162/imag_a_00104',
             'location': 'Section 3.2 "Experimental", Table 3, row "NEXI", column "t ex (ms)": "103.9 [100.3, 107.5]"'},
            {'value': 14.3e-3, 'unit': 's', 'printed': '14.3 [12.2, 16.3] ms', 'field_T': None,
             'species': 'human', 'method': 'NEXI_dot (NEXI with an immobile dot compartment), in vivo',
             'source_key': 'uhl2024', 'doi': '10.1162/imag_a_00104',
             'location': 'Section 3.2 "Experimental", Table 3, row "NEXI dot", '
                         'column "t ex (ms)": "14.3 [12.2, 16.3]"'},
            {'value': 2.90e-3, 'unit': 's', 'printed': '2.90 [2.71, 3.09] ms', 'field_T': None,
             'species': 'human', 'method': 'NEXI_dot,RM, in vivo', 'source_key': 'uhl2024',
             'doi': '10.1162/imag_a_00104',
             'location': 'Section 3.2 "Experimental", Table 3, row "NEXI dot, RM", '
                         'column "t ex (ms)": "2.90 [2.71, 3.09]". '
                         'More than ten times shorter than the default from the same data: the four model '
                         'variants disagree (Section 3.2: "The four model variants give very different '
                         'exchange time estimates.").'},
            {'value': None, 'range': (30e-3, 60e-3), 'unit': 's', 'printed': '30–60 ms', 'field_T': None,
             'species': 'rat', 'temperature_C': 38.0,
             'method': 'NEXI, in vivo, multi-b multi-t PGSE, Datasets 1-3', 'region': 'cortex',
             'source_key': 'jelescu2022', 'doi': '10.1016/j.neuroimage.2022.119277',
             'location': 'Section 3.2.2 "Experimental: in vivo rat GM": "For Datasets 1–3, the estimated exchange '
                         'time t ex was higher with NEXI than with t ex K(t): 30–60 ms vs 10–40 ms, respectively, '
                         'in cortex and 25–65 ms vs 10–25 ms, respectively, in hippocampus." Temperature: Section '
                         '2.2 "body temperature was monitored and maintained around 38 °C".',
             'derived_permeability': {
                 'by': 'the authors', 'value': None, 'range': (2.1e-6, 33e-6), 'unit': 'm/s',
                 'printed': 'P ∼ [2.1 − 33] × 10^−3 μm/ms',
                 'formula': 'P = d / (4 (t_ex − d^2 / (32 D_i,∥))), approximated by the authors as P ≅ d / (4 t_ex)',
                 'geometry': 'cylinder (neurite) of diameter d', 'radius_m': (0.25e-6, 1.0e-6),
                 'location': 'Section 3.2.2: "Assuming d ∼ [0.5 − 2] μm, a characteristic exchange time '
                             't ex = [15 − 60] ms, as estimated here using NEXI, yields P ∼ [2.1 − 33] × 10^−3 μm/ms."',
                 'note': 'Uses t_ex, which is (1 - f) tau_i, in place of tau_i.'}},
            {'value': None, 'range': (10e-3, 30e-3), 'unit': 's', 'printed': '10–30 ms', 'field_T': None,
             'species': 'rat', 'temperature_C': 38.0, 'method': 'NEXI, in vivo, Dataset 4 (x2.5 voxel volume)',
             'region': 'cortex and hippocampus', 'source_key': 'jelescu2022',
             'doi': '10.1016/j.neuroimage.2022.119277',
             'location': 'Section 3.2.2: "Remarkably, for Dataset 4 (x2.5 voxel volume compared to Datasets 1–3), '
                         'the two approaches displayed better agreement and estimates for t ex ranged 10–30 ms."'},
            {'value': 21e-3, 'unit': 's', 'printed': '21±4 ms', 'field_T': None, 'species': 'rat',
             'temperature_C': 38.0, 'method': 'Kärger-model fit of time-dependent mean kurtosis MK(t), in vivo',
             'region': 'cortex', 'source_key': 'jelescu2022', 'doi': '10.1016/j.neuroimage.2022.119277',
             'location': 'Section 3.1.2 "Exchange vs structural disorder: time-dependent diffusion and kurtosis": '
                         '"The GM yielded exchange times in the same range as the diffusion times explored and '
                         'thus with better precision, e.g., tex=21±4ms in cortex and tex=16±3ms in hippocampus."'},
        ],
        'citation': {
            'key': 'uhl2024',
            'authors': 'Uhl Q, Pavan T, Molendowska M, Jones DK, Palombo M, Jelescu IO',
            'title': 'Quantifying human gray matter microstructure using neurite exchange imaging (NEXI) '
                     'and 300 mT/m gradients',
            'journal': 'Imaging Neuroscience',
            'year': 2024,
            'doi': '10.1162/imag_a_00104',
        },
        'citations': {
            'jelescu2022': {
                'authors': 'Jelescu IO, de Skowronski A, Geffroy F, Palombo M, Novikov DS',
                'title': 'Neurite Exchange Imaging (NEXI): A minimal model of diffusion in gray matter with '
                         'inter-compartment water exchange',
                'journal': 'NeuroImage', 'year': 2022, 'doi': '10.1016/j.neuroimage.2022.119277'},
        },
        'description': 'Neurite / extra-neurite water exchange time t_ex = 1 / (r_ie + r_ei) of grey matter, '
                       'from fits of the anisotropic Kärger (NEXI) model. Not a permeability and not tau_i.',
        'note': 'Model-dependent: on one human dataset the NEXI variants span 2.9 ms to 104 ms (Uhl 2024 '
                'Table 3). The default is the variant with the lowest AICc. Human in vivo. Full texts read: '
                'Uhl 2024 and Jelescu 2022 (open access, PubMed Central).',
    },
    'exchange_rate_grey_matter': {
        'default': {
            'value': 1.2,
            'unit': 's^-1',
            'printed': '1.2 ± 0.6 s^-1 (mean ± SD over 2 x 18 measurements)',
            'field_T': None,
            'species': 'human',
            'temperature_C': None,   # in vivo; not stated
            'method': 'filter exchange imaging (FEXI), in vivo, 3 T, 18 healthy volunteers, test-retest',
            'region': 'lenticular nucleus (deep grey matter)',
            'source_key': 'lampinen2017',
            'doi': '10.1002/mrm.26195',
            'location': 'Results "Test-Retest Study", Table 2 ("FEXI Test-Retest Results In the 18 Healthy '
                        'Volunteers"), row "LN", column "AXR (s −1 )": "1.2 ± 0.6". Table note: "LN, '
                        'lenticular nucleus; TH, thalamus."',
        },
        'alternatives': [
            {'value': 0.8, 'unit': 's^-1', 'printed': '0.8 ± 0.4 s^-1', 'field_T': None, 'species': 'human',
             'method': 'FEXI, in vivo, 3 T', 'region': 'thalamus', 'source_key': 'lampinen2017',
             'doi': '10.1002/mrm.26195',
             'location': 'Results "Test-Retest Study", Table 2, row "TH", column "AXR (s −1 )": "0.8 ± 0.4"'},
            {'value': 110.0, 'unit': 's^-1', 'printed': '110 ± 20 s −1 at 25° C', 'field_T': None,
             'species': 'mouse (neonatal, excised spinal cord, fixed)', 'temperature_C': 25.0,
             'method': 'diffusion exchange spectroscopy (DEXSY), rapid four-point method, single-sided '
                       'static-gradient NMR', 'region': 'spinal cord ("contains mostly gray matter", Discussion)',
             'source_key': 'williamson2019', 'doi': '10.7554/eLife.51101',
             'location': 'Results, "Rapid exchange measurement agrees with full DEXSY": "Additionally, the value '
                         'from repeated measurements on five fixed samples was 110 ± 20 s −1 at 25° C (mean ± SD '
                         'taken across all 3 × 5 measurements)". The same section: "The AXR from three repeat '
                         'measurements was 110 ± 30 s −1."',
             'note': 'About 90 times the default. Fixed tissue at 25 °C, and the measurement is sensitive to '
                     'structures smaller than about 1400 nm (Discussion), so the restricted pool is not the '
                     'whole intracellular space; the two values are not the same quantity of the same tissue.'},
        ],
        'citation': {
            'key': 'lampinen2017',
            'authors': 'Lampinen B, Szczepankiewicz F, van Westen D, Englund E, Sundgren PC, Latt J, '
                       'Stahlberg F, Nilsson M',
            'title': 'Optimal experimental design for filter exchange imaging: Apparent exchange rate '
                     'measurements in the healthy brain and in intracranial tumors',
            'journal': 'Magnetic Resonance in Medicine',
            'year': 2017,
            'doi': '10.1002/mrm.26195',
        },
        'citations': {
            'williamson2019': {
                'authors': 'Williamson NH, Ravin R, Benjamini D, Merkle H, Falgairolle M, O\'Donovan MJ, '
                           'Blivis D, Ide D, Cai TX, Ghorashi NS, Bai R, Basser PJ',
                'title': 'Magnetic resonance measurements of cellular and sub-cellular membrane structures in '
                         'live and fixed neural tissue',
                'journal': 'eLife', 'year': 2019, 'doi': '10.7554/eLife.51101'},
        },
        'description': 'Apparent exchange rate (AXR) of grey matter: the apparent estimate of k = k_sf + k_fs '
                       'between the slow and fast diffusion components (Lampinen 2017, Theory). Not a permeability.',
        'note': 'Full texts read: Lampinen 2017 and Williamson 2019 (open access, PubMed Central).',
    },
    'exchange_rate_white_matter': {
        'default': {
            'value': 0.3,
            'unit': 's^-1',
            'printed': '0.3 ± 0.2 s^-1 (mean ± SD over 2 x 18 measurements)',
            'field_T': None,
            'species': 'human',
            'temperature_C': None,   # in vivo; not stated
            'method': 'filter exchange imaging (FEXI), in vivo, 3 T, 18 healthy volunteers, test-retest',
            'region': 'corpus callosum (genu and splenium)',
            'source_key': 'lampinen2017',
            'doi': '10.1002/mrm.26195',
            'location': 'Results "Test-Retest Study", Table 2 ("FEXI Test-Retest Results In the 18 Healthy '
                        'Volunteers"), row "CC", column "AXR (s −1 )": "0.3 ± 0.2" (CV 78 %). Table note: '
                        '"CC, genu and splenium of the corpus callosum".',
        },
        'alternatives': [
            {'value': 0.9, 'unit': 's^-1', 'printed': '0.9 ± 0.3 s^-1', 'field_T': None, 'species': 'human',
             'method': 'FEXI, in vivo, 3 T', 'region': 'anterior limb of the internal capsule',
             'source_key': 'lampinen2017', 'doi': '10.1002/mrm.26195',
             'location': 'Results "Test-Retest Study", Table 2, row "ALIC", column "AXR (s −1 )": "0.9 ± 0.3"'},
            {'value': 0.4, 'unit': 's^-1', 'printed': '0.4 ± 0.1 s^-1', 'field_T': None, 'species': 'human',
             'method': 'FEXI, in vivo, 3 T', 'region': 'corticospinal tract (printed "cerebrospinal tract")',
             'source_key': 'lampinen2017', 'doi': '10.1002/mrm.26195',
             'location': 'Results "Test-Retest Study", Table 2, row "CST", column "AXR (s −1 )": "0.4 ± 0.1"'},
            {'value': 1.8, 'unit': 's^-1', 'printed': '1.8 ± 0.3 s^-1', 'field_T': None, 'species': 'human',
             'method': 'FEXI, in vivo, 3 T', 'region': 'anterior corona radiata', 'source_key': 'lampinen2017',
             'doi': '10.1002/mrm.26195',
             'location': 'Results "Test-Retest Study", Table 2, row "ACR", column "AXR (s −1 )": "1.8 ± 0.3"'},
            {'value': None, 'unit': 's^-1', 'field_T': None, 'species': 'human',
             'method': 'FEXI, in vivo', 'region': 'internal capsule; frontal white matter',
             'source_key': 'nilsson2013', 'doi': '10.1002/mrm.24395',
             'location': 'NEEDS VERIFICATION',
             'note': 'Full text not reached; no value is entered from the abstract (the abstract is quoted in '
                     'kappa_membrane\'s location).'},
        ],
        'citation': {
            'key': 'lampinen2017',
            'authors': 'Lampinen B, Szczepankiewicz F, van Westen D, Englund E, Sundgren PC, Latt J, '
                       'Stahlberg F, Nilsson M',
            'title': 'Optimal experimental design for filter exchange imaging: Apparent exchange rate '
                     'measurements in the healthy brain and in intracranial tumors',
            'journal': 'Magnetic Resonance in Medicine',
            'year': 2017,
            'doi': '10.1002/mrm.26195',
        },
        'description': 'Apparent exchange rate (AXR) of white matter: the apparent estimate of k = k_sf + k_fs '
                       'between the slow and fast diffusion components (Lampinen 2017, Theory). Not a permeability.',
        'note': 'Varies six-fold across white-matter ROIs of one study (Table 2). Full text read: Lampinen 2017 '
                '(open access, PubMed Central).',
    },
    'intracellular_residence_time': {
        'default': {
            'value': 0.57,
            'unit': 's',
            'printed': '0.57(0.05) s (mean (SD) over the ROI)',
            'field_T': None,
            'species': 'human',
            'temperature_C': None,   # in vivo; not stated
            'cell_type': 'myelinated axon (white matter)',
            'method': 'random-forest regression trained on Monte Carlo simulations of permeable cylinders, '
                      'PGSTE protocol, in vivo, 3 T, healthy volunteers',
            'region': 'splenium of the corpus callosum (CC-S)',
            'source_key': 'nedjatigilani2017',
            'doi': '10.1016/j.neuroimage.2017.02.013',
            'location': 'Section 3.2 "Healthy subjects", '
                        'Table 3 ("Mean, standard deviation and coefficient of variation of all '
                        'parameters estimated using the random forest model in all six regions of interest."), '
                        'row "CC-S", column "τi (s)": "0.57(0.05)"; the same row gives α (μm) "1.65(0.58)". '
                        'Full text: the published version deposited in the Imperial College repository '
                        '(hdl.handle.net/10044/1/48363).',
            'derived_permeability': {
                'by': 'this table (not printed by the paper)', 'value': 1.447e-6, 'unit': 'm/s',
                'formula': 'kappa = R / (2 tau_i) (the paper\'s own relation, Section 2.2: "k = R/(2 τi)")',
                'geometry': 'cylinder, well-mixed interior',
                'radius_m': 1.65e-6,
                'note': 'R = the row\'s alpha, a volume-weighted radius index (Section 2.4), not a mean radius: '
                        '1.65e-6 / (2 x 0.57) = 1.447e-6 m/s.'},
        },
        'alternatives': [
            {'value': 0.60, 'unit': 's', 'printed': '0.60(0.04) s', 'field_T': None, 'species': 'human',
             'cell_type': 'myelinated axon (white matter)', 'method': 'as the default',
             'region': 'genu of the corpus callosum (CC-G)', 'source_key': 'nedjatigilani2017',
             'doi': '10.1016/j.neuroimage.2017.02.013',
             'location': 'Section 3.2 "Healthy subjects", Table 3, row "CC-G", column "τi (s)": "0.60(0.04)"'},
            {'value': 0.48, 'unit': 's', 'printed': '0.48(0.10) s', 'field_T': None, 'species': 'human',
             'cell_type': 'myelinated axon (white matter)', 'method': 'as the default',
             'region': 'left anterior limb of the internal capsule (ALIC-L)', 'source_key': 'nedjatigilani2017',
             'doi': '10.1016/j.neuroimage.2017.02.013',
             'location': 'Section 3.2 "Healthy subjects", Table 3, row "ALIC-L", column "τi (s)": "0.48(0.10)"'},
            {'value': 0.46, 'unit': 's', 'printed': '0.46(0.11) s', 'field_T': None, 'species': 'human',
             'cell_type': 'myelinated axon (white matter)', 'method': 'as the default',
             'region': 'right anterior limb of the internal capsule (ALIC-R)', 'source_key': 'nedjatigilani2017',
             'doi': '10.1016/j.neuroimage.2017.02.013',
             'location': 'Section 3.2 "Healthy subjects", Table 3, row "ALIC-R", column "τi (s)": "0.46(0.11)"'},
            {'value': 0.54, 'unit': 's', 'printed': '0.54(0.08) s', 'field_T': None, 'species': 'human',
             'cell_type': 'myelinated axon (white matter)', 'method': 'as the default',
             'region': 'left corticospinal tract (CST-L)', 'source_key': 'nedjatigilani2017',
             'doi': '10.1016/j.neuroimage.2017.02.013',
             'location': 'Section 3.2 "Healthy subjects", Table 3, row "CST-L", column "τi (s)": "0.54(0.08)"'},
            {'value': 0.52, 'unit': 's', 'printed': '0.52(0.09) s', 'field_T': None, 'species': 'human',
             'cell_type': 'myelinated axon (white matter)', 'method': 'as the default',
             'region': 'right corticospinal tract (CST-R)', 'source_key': 'nedjatigilani2017',
             'doi': '10.1016/j.neuroimage.2017.02.013',
             'location': 'Section 3.2 "Healthy subjects", Table 3, row "CST-R", column "τi (s)": "0.52(0.09)"'},
            {'value': 0.75, 'unit': 's', 'printed': '0.75 ± 0.05 s', 'field_T': None,
             'species': 'rat (cultured cortical neurons on microbeads)', 'temperature_C': 36.5,
             'cell_type': 'neuron', 'method': 'inversion-recovery spin echo (IRSE) T1 relaxometry of perfused '
             'microbead-adherent cells, slow-exchange regime', 'source_key': 'yang2018',
             'doi': '10.1002/mrm.26781',
             'location': 'Results, "τi under Normal Conditions": "The estimated τ i is 0.75 ± 0.05 s for neurons '
                         '( n measurement = 12, n pup = 6) and 0.57 ± 0.03 s for astrocytes ( n measurement = 21, '
                         'n pup = 12)". Temperature: Methods, "IRSE Experiments with Perfused Microbead-Adherent '
                         'Cells": "Tyrode\'s solution (media) was perfused through the sample at 36.5 °C".',
             'derived_permeability': {
                 'by': 'the authors', 'value': 2e-6, 'unit': 'm/s', 'printed': '∼2 × 10^-3 mm/s',
                 'formula': 'P_d = V_i / (A τ_i) (Eq. [1]); for a sphere V_i / A = R / 3',
                 'geometry': 'sphere ("∼10-μm diameter spherical cell bodies", dendrites ignored)',
                 'radius_m': 5e-6,
                 'location': 'Discussion: "Making the very rough approximation that neurons can be considered as '
                             'having ∼10-μm diameter spherical cell bodies, as suggested by Fig. 2 , and ignoring '
                             'dendritic branching, the diffusional water permeability P d is ∼2 × 10 -3 mm/s."'}},
            {'value': 0.57, 'unit': 's', 'printed': '0.57 ± 0.03 s', 'field_T': None,
             'species': 'rat (cultured cortical astrocytes on microbeads)', 'temperature_C': 36.5,
             'cell_type': 'astrocyte', 'method': 'as the neuron row', 'source_key': 'yang2018',
             'doi': '10.1002/mrm.26781',
             'location': 'Results, "τi under Normal Conditions" (the sentence quoted on the neuron row).'},
            {'value': None, 'range': (3.8e-3, 4.5e-3), 'unit': 's', 'printed': '3.8–4.5 ms', 'field_T': None,
             'species': 'rat (perfusion-fixed brain, ex vivo)', 'temperature_C': 37.0, 'cell_type': 'neurite',
             'method': 'SMEX model fit (stick-Gaussian Kärger exchange), PGSE at 16.4 T, b up to 100 ms/um^2',
             'region': 'cortex (sub-ROIs)', 'source_key': 'olesen2022', 'doi': '10.1016/j.neuroimage.2022.118976',
             'location': 'Table 1 ("Model parameter estimates given as min-max ranges from fitting to each sub-ROI '
                         'in the cortex"), row "SMEX", column "τ n" (ms): "3.8–4.5". Temperature: Section 2.5 '
                         '"placed in a 5 mm NMR tube with Fluorinert kept at 37 ᵒC".',
             'note': 'More than a hundred times shorter than the default: fixed tissue (Section 4.3 discusses '
                     'fixation raising permeability), grey matter, and a different model.',
             'derived_permeability': {
                 'by': 'the authors', 'value': 125e-6, 'unit': 'm/s', 'printed': '125 μm/s',
                 'formula': 'kappa = R / (2 tau_n) (well-mixed cylinder; "relation to exchange rate and surface to '
                            'volume ratio")', 'geometry': 'cylinder (neurite)', 'radius_m': 1e-6,
                 'location': 'Section 4.3: "for a typical neurite radius of 1 μm ( Palombo et al., 2021 ) with '
                             'residence time 4 ms, this results in a permeability of 125 μm/s."'}},
        ],
        'citation': {
            'key': 'nedjatigilani2017',
            'authors': 'Nedjati-Gilani GL, Schneider T, Hall MG, Cawley N, Hill I, Ciccarelli O, Drobnjak I, '
                       'Wheeler-Kingshott CAMG, Alexander DC',
            'title': 'Machine learning based compartment models with permeability for white matter '
                     'microstructure imaging',
            'journal': 'NeuroImage',
            'year': 2017,
            'doi': '10.1016/j.neuroimage.2017.02.013',
        },
        'citations': {
            'yang2018': {
                'authors': 'Yang DM, Huettner JE, Bretthorst GL, Neil JJ, Garbow JR, Ackerman JJH',
                'title': 'Intracellular water preexchange lifetime in neurons and astrocytes',
                'journal': 'Magnetic Resonance in Medicine', 'year': 2018, 'doi': '10.1002/mrm.26781'},
            'olesen2022': {
                'authors': 'Olesen JL, Østergaard L, Shemesh N, Jespersen SN',
                'title': 'Diffusion time dependence, power-law scaling, and exchange in gray matter',
                'journal': 'NeuroImage', 'year': 2022, 'doi': '10.1016/j.neuroimage.2022.118976'},
        },
        'description': 'Mean residence time tau_i of water inside a cell (cell type in each source\'s '
                       '\'cell_type\'). kappa = R / (2 tau_i) for a cylinder, R / (3 tau_i) for a sphere, '
                       'both for a well-mixed interior.',
        'note': 'Spans 3.8 ms (fixed rat neurites) to 0.75 s (cultured rat neurons). Full texts read: '
                'Nedjati-Gilani 2017 (published version, Imperial College repository), Yang 2018 and Olesen '
                '2022 (PubMed Central).',
    },
    'kappa_membrane_direct': {
        'default': {
            'value': 3.4e-6,
            'unit': 'm/s',
            'printed': '3.4 x 10^-4 cm/sec (Pd at 25 °C, one axon)',
            'field_T': None,
            'species': 'squid (Loligo pealii), giant axon',
            'temperature_C': 25.0,
            'cell_type': 'unmyelinated axon (squid giant axon)',
            'method': 'tritiated-water (HTO) efflux from a single axon; Pd from the efflux half-time and the '
                      'optically measured axon radius b (Methods: for a cylinder "the V_i/A_m ratio becomes half '
                      'the radius")',
            'source_key': 'nevis1958',
            'doi': '10.1085/jgp.41.5.927',
            'location': 'Results B, "Squid giant axon water fluxes", Table VII (Pd x 10^4 cm/sec by temperature): '
                        'the 25 °C row (one axon) "3.4"; Table V, axon VIa at 25 °C, "3.4". The abstract states '
                        '"Pd measurements were made for squid giant axons at 25°C. and found to yield a value of '
                        '4 x 10–4 cm.–1 sec.–1." Read from the PubMed Central record of the scanned article '
                        '(text layer of the page scans); the table columns were checked by recomputing each '
                        'row of Table VII from the per-axon Pd of Table V.',
        },
        'alternatives': [
            {'value': 2.5e-6, 'unit': 'm/s', 'printed': '2.5 ± 0.15 x 10^-4 cm/sec (4 axons)', 'field_T': None,
             'species': 'squid (Loligo pealii), giant axon', 'temperature_C': 13.0,
             'cell_type': 'unmyelinated axon (squid giant axon)', 'method': 'as the default',
             'source_key': 'nevis1958', 'doi': '10.1085/jgp.41.5.927',
             'location': 'Table VII, the 13 °C row (4 axons): "2.5 ± 0.15" (x 10^-4 cm/sec); Table V axons at 13 °C: '
                         '2.5, 2.6, 2.4, 2.3.'},
            {'value': 4.8e-6, 'unit': 'm/s', 'printed': '4.8 ± 1.1 x 10^-4 cm/sec (3 axons)', 'field_T': None,
             'species': 'squid (Loligo pealii), giant axon', 'temperature_C': 30.0,
             'cell_type': 'unmyelinated axon (squid giant axon)', 'method': 'as the default',
             'source_key': 'nevis1958', 'doi': '10.1085/jgp.41.5.927',
             'location': 'Table VII, the 30 °C row (3 axons): "4.8 ± 1.1" (x 10^-4 cm/sec); Table V axons at 30 °C: '
                         '3.9, 4.3, 6.2.'},
        ],
        'citation': {
            'key': 'nevis1958',
            'authors': 'Nevis AH',
            'title': 'Water transport in invertebrate peripheral nerve fibers',
            'journal': 'Journal of General Physiology',
            'year': 1958,
            'doi': '10.1085/jgp.41.5.927',
        },
        'description': 'Diffusional water permeability of a cell membrane measured directly (tracer flux), '
                       'not inferred from a diffusion-MRI exchange model.',
        'note': 'No direct measurement of a mammalian neuronal, glial or axonal membrane was verified in full '
                'text; the permeabilities derived by authors from residence times (Yang 2018, Olesen 2022) and '
                'from exchange times (Jelescu 2022) are stored, labelled derived, under '
                'intracellular_residence_time and exchange_time_grey_matter. The default is not a human value '
                '(none verified). It is the 25 °C row of Nevis 1958 Table VII, which is ONE axon; the 13 °C row '
                '(four axons, 2.5e-6 m/s) and the 30 °C row (three axons, 4.8e-6 m/s) are the alternatives.',
    },
    'rho2_axon_membrane': {
        'default': {
            'value': 1.16e-6,
            'unit': 'm/s',
            'field_T': None,   # surface relaxivity is a membrane material property, treated field-independent (calibrated ex vivo)
            'species': 'human',
            'method': 'diffusion-relaxation MRI calibrated against histology (ex vivo corpus callosum)',
            'source_key': 'barakovic2023',
            'location': 'Brownstein-Tarr surface relaxivity fit; 1/T2a = 1/T2c + 2*rho2/r, '
                        'rho2 ~ 1.16 nm/ms calibrated to inner axon radius from histology. '
                        'Note: T2_bulk_intra (cytoplasm) from same study = 127 ms ex vivo; '
                        'in vivo 3T use T2_apparent=50ms (Table 2) and correct: '
                        'T2_bulk_intra = 1/(1/50ms - 2*rho2/R_mean) ~ 54 ms for R=1.46um.',
        },
        'alternatives': [
            {
                'value': 3.7e-6,
                'unit': 'm/s',
                'field_T': None,
                'note': 'Hollow polymer phantom fibers (not brain tissue)',
                'source_key': 'canalesrodriguez2024',
                'location': 'Phantom calibration of transverse surface relaxivity '
                            'in hollow polymer microfibers (non-tissue reference).',
            }
        ],
        'citation': {
            'key': 'barakovic2023',
            'authors': 'Barakovic M, Pizzolato M, Tax CMW, Rudrapatna U, Magon S, '
                       'Dyrby TB, Granziera C, Thiran JP, Jones DK, Canales-Rodriguez EJ',
            'title': 'Estimating axon radius using diffusion-relaxation MRI: calibrating '
                     'a surface-based relaxation model with histology',
            'journal': 'Frontiers in Neuroscience',
            'year': 2023,
            'doi': '10.3389/fnins.2023.1209521',
        },
        'description': 'T2 surface relaxivity of the axon membrane (axolemma). '
                       'Use paired with corrected T2_bulk_intra, not T2_apparent.',
    },
    'axon_radius_mean': {
        'default': {
            'value': 0.305e-6,
            'unit': 'm',
            'field_T': None,
            'species': 'human',
            'method': 'electron microscopy of corpus callosum',
            'source_key': 'aboitiz1992',
            'location': 'Table 2, overall mean fiber diameter ~0.6 um (radius ~0.3 um) '
                        'for thin fibers; up to ~6 um diameter for large fibers. '
                        '[CORRECTED 2026-05-31: prior value 3e-6 m was a 10x error '
                        '(it stated a 3 um radius / 6 um diameter, contradicting its '
                        'own note of radius ~0.3 um). Now consistent with '
                        'gamma_shape_diameter=2.0, scale=0.304 um (mean radius 0.30 um).]',
        },
        'alternatives': [],
        'citation': _CITATION_ABOITIZ1992,
        'description': 'Mean axon radius in human corpus callosum',
        'note': 'Axon radius varies strongly by tract region (genu vs splenium) '
                'and species',
    },
    'axon_radius_std': {
        'default': {
            'value': 0.215e-6,
            'unit': 'm',
            'field_T': None,
            'species': 'human',
            'method': 'electron microscopy of corpus callosum',
            'source_key': 'aboitiz1992',
            'location': 'Table 2, standard deviation of fiber diameter distribution '
                        'across callosal regions. [CORRECTED 2026-05-31 to match '
                        'gamma_shape_diameter=2.0, scale=0.304 um: std radius = '
                        '0.5*sqrt(alpha)*scale = 0.215 um. Prior value 1e-6 m paired '
                        'with the erroneous 3e-6 mean.]',
        },
        'alternatives': [],
        'citation': _CITATION_ABOITIZ1992,
        'description': 'Standard deviation of axon radius in corpus callosum',
    },
    'g_ratio_typical': {
        'default': {
            'value': 0.7,
            'unit': '',
            'field_T': None,
            'species': 'human',
            'method': 'combined MRI + histology',
            'source_key': 'stikov2015',
            'location': 'Section "Results", mean g-ratio = 0.70 +/- 0.03 in CC; '
                        'Figure 4, histogram of g-ratio values peaked at ~0.7',
        },
        'alternatives': [],
        'citation': {
            'key': 'stikov2015',
            'authors': 'Stikov N, Campbell JS, Stroh T, et al.',
            'title': 'In vivo histology of the myelin g-ratio with magnetic resonance imaging',
            'journal': 'NeuroImage',
            'year': 2015,
            'doi': '10.1016/j.neuroimage.2015.05.023',
        },
        'description': 'Typical g-ratio (inner/outer myelin radius) in WM',
    },
    'myelin_water_proton_density': {
        'default': {
            'value': 0.40,
            'unit': '(fraction of myelin-sheath VOLUME that is water)',
            'field_T': None,
            'species': 'human',
            'method': 'myelin volume fraction imaging (MRI vs histology)',
            'source_key': 'west2018',
            'location': 'Myelin water content ~40% by volume of the myelin sheath; the '
                        'remainder (~60%) is dry lipid + protein. This is the INTRINSIC, '
                        'per-unit-volume water/proton content that weights the myelin spin '
                        'population (the proton-density weight for myelin walkers/spins).',
        },
        'alternatives': [],
        'citation': _CITATION_WEST2018,
        'description': 'Myelin water content: the fraction of the myelin-sheath VOLUME that '
                       'is water (~0.40 = 1 - lipid/protein fraction). This is an INTENSIVE, '
                       'per-volume proton-density weight -- NOT a signal fraction.',
        # The three myelin water quantities, kept distinct on purpose:
        #   myelin_water_proton_density        ~0.40  water per unit myelin volume (THIS; an input)
        #   myelin (geometric) vol frac ~0.375 myelin sheath vol / total (from g-ratio+packing)
        #   myelin_water_fraction (MWF) ~0.15  measured short-T2 signal fraction (DERIVED)
        # Chain:  MWF ~= f_myelin_vol * myelin_water_proton_density  ->  ~0.375 * 0.40 ~= 0.15,
        # i.e. every unit of myelin water SIGNAL corresponds to ~2.5 (=1/0.40) units of
        # myelin geometric VOLUME.  Use THIS (0.40) as the per-volume weight; never the 0.15.
        'note': 'Do NOT use the MWF value (myelin_water_fraction ~0.15, a measured signal '
                'fraction) as a per-volume weight. This (=0.40, West 2018) is the water '
                'content per myelin volume; the sheath is ~60% dry lipid/protein.',
    },
    'myelin_water_fraction': {
        'default': {
            'value': 0.15,
            'unit': '',
            'field_T': 1.5,
            'species': 'human',
            'method': 'multi-component T2 relaxometry (MESE, NNLS decomposition)',
            'source_key': 'mackay1994',
            'location': 'Section "Results", myelin water fraction in normal-appearing WM '
                        '~11-15% depending on region; Figure 5',
        },
        'alternatives': [
            {
                # 3T GRASE, n=31 healthy volunteers, 25 bilateral ROIs.
                # Verbatim (Table 1): GCC "9.59 ± 1.83", SCC "11.81 ± 1.38",
                # BCC "9.61 ± 1.70", PLIC "16.04 ± 1.36", SLF "10.75 ± 1.52" (all %).
                # Recommended 3T reference: larger sample, region-specific values.
                'value': 0.096,        # CC genu; use 0.118 for CC splenium
                'value_cc_splenium': 0.118,
                'unit': '',
                'field_T': 3.0,
                'species': 'human',
                'method': '3D GRASE multi-component T2 relaxometry, 3T Philips Achieva',
                'source_key': 'uddin2019',
                'location': 'Table 1, MWF column (%), 31 healthy volunteers ages 18-57; '
                            'GCC=9.59±1.83%, SCC=11.81±1.38%, BCC=9.61±1.70%, '
                            'PLIC=16.04±1.36%, SLF=10.75±1.52%',
                'citation': {
                    'key': 'uddin2019',
                    'authors': 'Uddin MN, Figley TD, Solar KG, Shatil AS, Figley CR',
                    'title': 'Comparisons between multi-component myelin water fraction, '
                             'T1w/T2w ratio, and diffusion tensor imaging measures in '
                             'healthy human brain structures',
                    'journal': 'Scientific Reports',
                    'year': 2019,
                    'doi': '10.1038/s41598-019-39199-x',
                    'pmcid': 'PMC6384876',
                },
            },
            {
                # 3T MESE 32-echo, n=14 healthy controls.
                # Verbatim (abstract): "In the HC group, we found a mean MWF in WM
                # of 0.15 ± 0.058 over all defined ROIs."
                # Table 2 regional values: CST ~0.21, frontal WM ~0.14, CC genu 0.089,
                # CC splenium 0.104.
                'value': 0.15,
                'unit': '',
                'field_T': 3.0,
                'species': 'human',
                'method': 'multi-echo spin echo (MESE) 32-echo, 3T Siemens',
                'source_key': 'faizy2016',
                'location': 'Abstract: "In the HC group, we found a mean MWF in WM of '
                            '0.15 ± 0.058 over all defined ROIs." Table 2: CC genu '
                            '0.089±0.051, CC splenium 0.104±0.034, CST ~0.21, '
                            'frontal WM ~0.14; 14 healthy controls',
                'citation': {
                    'key': 'faizy2016',
                    'authors': 'Faizy TD, Thaler C, Kumar D, Sedlacik J, Broocks G, '
                               'Grosser M, et al.',
                    'title': 'Heterogeneity of Multiple Sclerosis Lesions in Multislice '
                             'Myelin Water Imaging',
                    'journal': 'PLOS ONE',
                    'year': 2016,
                    'doi': '10.1371/journal.pone.0151496',
                    'pmcid': 'PMC4798764',
                },
            },
            {
                # 3T STAIR-EPI (T1-suppression method, NOT T2 spectrum decomposition).
                # aMWF = apparent MWF from short-T1 component suppression.
                # Verbatim (abstract): "NWM (10 ± 1.3%) in healthy volunteers"
                # Note: STAIR aMWF ~10% is systematically lower than MESE/GRASE ~12-15%
                # because it uses a different contrast mechanism (T1-based, not T2-based).
                # Not directly interchangeable with MESE/GRASE values.
                'value': 0.10,
                'unit': '',
                'field_T': 3.0,
                'species': 'human',
                'method': 'STAIR-EPI (short-TR adiabatic inversion recovery + EPI), 3T GE MR750; '
                          'apparent MWF, T1-suppression contrast — NOT equivalent to T2-spectrum MWF',
                'source_key': 'shaterian2023',
                'location': 'Abstract, Results: "NWM (10 ± 1.3%) in healthy volunteers"; '
                            'n=7 healthy volunteers, normal white matter (NWM); '
                            'centrum semiovale, subcortical WM, periventricular regions, CC',
                'citation': {
                    'key': 'shaterian2023',
                    'authors': 'Shaterian Mohammadi H, Moazamian D, Athertya JS, Shin SH, '
                               'Lo J, Suprana A, Malhi BS, Ma Y',
                    'title': 'Quantitative myelin water imaging using short TR adiabatic '
                             'inversion recovery prepared echo-planar imaging (STAIR-EPI) sequence',
                    'journal': 'Frontiers in Radiology',
                    'year': 2023,
                    'doi': '10.3389/fradi.2023.1263491',
                    'pmcid': 'PMC10568074',
                },
            },
        ],
        'citation': _CITATION_MACKAY1994,
        'description': 'Typical myelin water fraction in normal-appearing WM. '
                       'CAUTION: values differ by method (~10% STAIR, ~10-15% MESE/GRASE '
                       'at 3T) and by region (CST ~0.21 vs CC genu ~0.09-0.15). '
                       'mcDESPOT yields ~2x higher values than MESE (known positive bias).',
    },

    # ── Susceptibility ────────────────────────────────────────────────────────


    'g_ratio_corpus_callosum': {
        'default': {
            'value': 0.70,
            'unit': 'dimensionless (inner/outer radius)',
            'field_T': None,
            'species': 'human',
            'method': 'electron microscopy + g-ratio mapping',
            'source_key': 'stikov2015',
            'location': 'Fig. 4: mean g-ratio = 0.70 ± 0.02 across CC subregions',
        },
        'alternatives': [
            {
                'value': 0.68,
                'unit': 'dimensionless',
                'field_T': None,
                'species': 'human',
                'method': 'EM morphometry, corpus callosum genu',
                'source_key': 'aboitiz1992',
                'location': 'Table 1',
                'citation': _CITATION_ABOITIZ1992,
            },
        ],
        'citation': {
            'key': 'stikov2015',
            'authors': 'Stikov N, Campbell JSW, Stroh T, Lavelée M, Frey S, '
                       'Novek J, Nuber S, Ho MK, Bedell BJ, Dougherty RF, '
                       'Leppert IR, Boudreau M, Narayanan S, Duval T, Cohen-Adad J, '
                       'Gasecka A, Côté D, Pike GB',
            'title': 'In vivo histology of the myelin g-ratio with magnetic resonance imaging',
            'journal': 'NeuroImage',
            'year': 2015,
            'doi': '10.1016/j.neuroimage.2015.05.023',
        },
        'description': (
            'Mean g-ratio (inner axon radius / outer myelin radius) in human '
            'corpus callosum.  Relatively conserved across WM tracts in healthy '
            'adults (range ~0.65–0.75).  Used as canonical value in the hollow-'
            'cylinder susceptibility model: Δχ_a·(1−g²) determines the dipolar '
            'field amplitude.'
        ),
        'note': (
            'g-ratio varies with myelination state: decreases in remyelination '
            '(thicker myelin), increases in demyelination (thinner myelin).  '
            'MWF ≈ (1−g²)·f_axon, so g can be estimated from CPMG + DWI jointly.'
        ),
    },

    # ── Canonical packed-substrate parameters (shared by dmipy-sim MC and the
    #    analytical UnifiedWhiteMatterModel; see architecture/WHITE_MATTER_MODEL_DESIGN.md) ──
    'gamma_shape_diameter': {
        'default': {
            'value': 2.0,
            'unit': 'dimensionless (Gamma shape α over diameter)',
            'field_T': None,
            'species': 'human',
            'method': 'Gamma fit to fibre-diameter histogram',
            'source_key': 'aboitiz1992',
            'location': 'Gamma(α, scale) fit to corpus-callosum FIBRE (outer) diameters; '
                        'α=2.0 with scale 0.304 µm gives mean OUTER diameter 0.61 µm '
                        '(std 0.43 µm, mean radius 0.30 µm), the canonical packed '
                        'substrate. The right-skewed sub-micron shape (CV≈0.71) '
                        'matches the Aboitiz CC histogram. '
                        '[CORRECTED 2026-05-31: prior α=9.62 gave mean diameter '
                        '2.9 µm — a near-monodisperse large-caliber population that '
                        'contradicts the cited Aboitiz mean ~0.6 µm and broke the '
                        'extra-axonal tortuosity regime; see '
                        'architecture/WHITE_MATTER_MODEL_DESIGN.md parity log.]',
        },
        'alternatives': [],
        'citation': _CITATION_ABOITIZ1992,
        'description': 'Gamma shape parameter of the fibre (OUTER) diameter '
                       'distribution used in the canonical packed-cylinder substrate.',
        'note': 'Pairs with gamma_scale_diameter (OUTER/fibre diameter). mean_d = α·scale; '
                'std_d = sqrt(α)·scale. Inner (axon) diameter = g·outer. Surface-to-volume '
                'of the extra-axonal space (hence surface relaxivity) is set by this '
                'distribution.',
    },
    'gamma_scale_diameter': {
        'default': {
            'value': 0.304e-6,
            'unit': 'm (Gamma scale over diameter)',
            'field_T': None,
            'species': 'human',
            'method': 'Gamma fit to fibre-diameter histogram',
            'source_key': 'aboitiz1992',
            'location': 'scale (β_d) = 0.304 µm with α=2.0 → mean OUTER (fibre) '
                        'diameter 0.61 µm; inner (axon) = g·outer.',
        },
        'alternatives': [],
        'citation': _CITATION_ABOITIZ1992,
        'description': 'Gamma scale parameter of the canonical fibre (OUTER) '
                       'diameter distribution.',
        'note': 'Diameter-space scale of the OUTER (fibre) diameter -- the quantity '
                'histology reports (Aboitiz 1992, "Fiber composition..."). Outer '
                'radius = diameter/2; inner (axon) radius = g_ratio * outer radius.',
    },
    'D_extra_axonal_intrinsic': {
        'default': {
            'value': 1.7e-9,
            'unit': 'm^2/s',
            'field_T': None,
            'species': 'human',
            'method': 'intrinsic bulk diffusivity (tortuosity emerges from packing)',
            'source_key': 'beaulieu2002',
            'location': 'Intrinsic extra-axonal diffusivity equals the free '
                        'cytoplasmic value; the apparent perpendicular reduction '
                        'is a geometric (tortuosity) effect, not an intrinsic one.',
        },
        'alternatives': [],
        'citation': {
            'key': 'beaulieu2002',
            'authors': 'Beaulieu C',
            'title': 'The basis of anisotropic water diffusion in the nervous system',
            'journal': 'NMR in Biomedicine',
            'year': 2002,
            'doi': '10.1002/nbm.782',
        },
        'description': 'Intrinsic (pre-tortuosity) extra-axonal diffusivity used by '
                       'the Monte Carlo substrate, where the apparent perpendicular '
                       'reduction emerges from the explicit packed geometry.',
        'note': 'PARITY-CRITICAL. The Monte Carlo engine uses this intrinsic value '
                'and recovers the hindered λ_perp from packing; the analytical '
                'Zeppelin must instead apply the tortuosity λ_perp = λ_par·(1−v_ic) '
                'to match. The standalone (apparent) value D_extra_axonal=1.2e-9 is '
                'the already-tortuosity-reduced number appropriate for a Zeppelin '
                'fitted without an explicit substrate. Do not mix the two.',
    },
}


def get_constant(name):
    """Return a biophysical constant entry by name."""
    if name not in BIOPHYSICAL_CONSTANTS:
        raise KeyError(
            "Unknown biophysical constant '{}'. Available: {}".format(
                name, list(BIOPHYSICAL_CONSTANTS.keys())))
    return BIOPHYSICAL_CONSTANTS[name]


def get_default_value(name):
    """Return the default scalar value for a biophysical constant."""
    return get_constant(name)['default']['value']


def get_value(name, field_T=None, *, allow_nearest=False):
    """Return the scalar value for a constant, field-strength matched.

    If ``field_T`` is None, returns the ``default`` value.  Otherwise returns the
    value whose ``field_T`` matches exactly (searching the ``default`` first, then
    the ``alternatives`` list).  Field-independent constants (``field_T`` is None in
    their entry) always return the default regardless of the requested field.

    A requested ``field_T`` with no exact match for a field-DEPENDENT constant is an
    error: this function does NOT silently fall back to the default.  Such a silent
    fallback previously returned the 1.5 T default at 7 T (the longest T2 where the
    shortest was wanted), corrupting every high-field result without warning.  By
    default we now raise ``ValueError`` listing the available fields; pass
    ``allow_nearest=True`` to opt into the nearest available field with a warning.

    This is the single accessor the canonical :class:`UnifiedWhiteMatterParameters`
    uses to pick field-matched relaxation values (e.g. 3 T Barakovic T2 vs the
    1.5 T MacKay defaults).
    """
    entry = get_constant(name)
    default = entry['default']
    # field-independent constant, or no specific field requested -> the default
    if field_T is None or default.get('field_T') is None:
        return default['value']
    candidates = [default] + list(entry.get('alternatives', []))
    for c in candidates:                      # exact field match
        if c.get('field_T') == field_T:
            return c['value']
    available = sorted({c['field_T'] for c in candidates if c.get('field_T') is not None})
    if allow_nearest and available:
        nearest = min(available, key=lambda f: abs(f - field_T))
        value = next(c['value'] for c in candidates if c.get('field_T') == nearest)
        warnings.warn(
            f"get_value('{name}', field_T={field_T}): no exact field match; using the "
            f"nearest available field {nearest} T (available: {available}). Add a "
            f"field-matched, cited entry for {field_T} T to silence this.",
            stacklevel=2)
        return value
    raise ValueError(
        f"get_value('{name}', field_T={field_T}): no field-matched value "
        f"(available fields: {available}). Tissue relaxation is field-dependent, so "
        f"silently using another field's value would bias results -- add a cited "
        f"entry for {field_T} T, pass field_T=None for the field-independent default, "
        f"or pass allow_nearest=True to accept the nearest field with a warning.")


def canonical_white_matter(field_T=3.0):
    """Curated, field-matched white-matter constant set.

    Returns a flat dict of the physical ground-truth values for the canonical
    ``UnifiedWhiteMatterModel`` (shared by dmipy-fit analytical and dmipy-sim Monte
    Carlo).  Every value is sourced from :data:`BIOPHYSICAL_CONSTANTS` so the
    citations and alternatives travel with it.  ``field_T`` selects the
    relaxation values (default 3 T, the field used throughout the
    coherence-pathway paper).

    Notes
    -----
    * ``D_extra`` is the *intrinsic* (pre-tortuosity) diffusivity — the Monte
      Carlo value.  The analytical Zeppelin must apply
      ``lambda_perp = lambda_par * (1 - v_ic)`` to match (see
      ``D_extra_axonal_intrinsic`` note).
    * ``T2_*`` are field-matched bulk values; surface relaxivity ``rho2`` is
      paired with them so the apparent T2 is recovered via
      ``1/T2_app = 1/T2_bulk + 2*rho2/r``.  Do not double-count.
    """
    return {
        # geometry
        'gamma_shape_diameter': get_value('gamma_shape_diameter'),
        'gamma_scale_diameter': get_value('gamma_scale_diameter'),
        'g_ratio': get_value('g_ratio_corpus_callosum'),
        # myelin water content (per-volume proton-density weight ~0.40; NOT the MWF signal)
        'myelin_water_proton_density': get_value('myelin_water_proton_density'),
        # diffusivity (intrinsic; tortuosity is geometric)
        'D_intra': get_value('D_intra_axonal'),
        'D_extra': get_value('D_extra_axonal_intrinsic'),
        'D_csf': get_value('D_csf'),
        # relaxation (field-matched)
        # Relaxation/membrane are field-dependent; allow_nearest=True makes a missing
        # field-matched entry warn loudly and use the nearest field (never a silent
        # fallback). Add a cited entry at a new field to silence the warning.
        'T2_intra': get_value('T2_intra_axonal', field_T, allow_nearest=True),
        'T2_extra': get_value('T2_extra_axonal', field_T, allow_nearest=True),
        'T2_myelin': get_value('T2_myelin', field_T, allow_nearest=True),
        'T2_csf': get_value('T2_csf', field_T, allow_nearest=True),
        # longitudinal: read by any C1 replay that turns per-compartment T1 (the pack's
        # `per_comp.T1`), and by the two-pool MT descriptor's bound-pool recovery
        'T1_intra': get_value('T1_intra_axonal', field_T, allow_nearest=True),
        'T1_extra': get_value('T1_extra_axonal', field_T, allow_nearest=True),
        'T1_myelin': get_value('T1_myelin', field_T, allow_nearest=True),
        'T1_csf': get_value('T1_csf', field_T, allow_nearest=True),
        # membrane / surface
        'rho2': get_value('rho2_axon_membrane', field_T, allow_nearest=True),
        'rho1': get_value('rho1_axon_membrane', field_T, allow_nearest=True),
        # myelin is a radially-layered lamellar stack, so diffusion within it is anisotropic
        'D_myelin_radial': get_value('D_myelin_radial'),
        'D_myelin_tangential': get_value('D_myelin_tangential'),
        # C3: myelin's susceptibility anisotropy -- the delta_chi_a knob of the field basis
        'delta_chi_a': get_value('delta_chi_a_myelin'),
        'chi_iso_myelin': get_value('chi_iso_myelin'),
        'gamma_proton': get_value('gamma_proton'),
        'kappa': get_value('kappa_membrane', field_T, allow_nearest=True),
        # quantitative MT (two-pool). These are the MEASURED qMT observables, not the walk's
        # (kappa_MT, dwell_time) -- kappa_MT depends on the substrate's own S/V, so it cannot live in a
        # tissue table. Convert with dmipy_analytic.relaxation.kappa_MT_from_forward_rate / dmipy_analytic.relaxation.dwell_time_from_fraction, or
        # Substrate.with_mt(f_bound, k_forward, S_over_V).
        'mt_bound_pool_fraction': get_value('mt_bound_pool_fraction', field_T, allow_nearest=True),
        'mt_exchange_rate': get_value('mt_exchange_rate', field_T, allow_nearest=True),
        'T2_bound': get_value('T2_bound_pool', field_T, allow_nearest=True),
    }
