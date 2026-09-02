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
"""

import warnings

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


_CITATION_STANISZ2005 = {
    'key': 'stanisz2005',
    'authors': 'Stanisz GJ, Odrobina EE, Pun J, Escaravage M, Graham SJ, Bronskill MJ, Henkelman RM',
    'title': 'T1, T2 relaxation and magnetization transfer in tissue at 3T',
    'journal': 'Magnetic Resonance in Medicine',
    'year': 2005,
    'doi': '10.1002/mrm.20605',
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
            'value': 267.513e6,
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
    # f_bound = k_f/(k_f + k_r) recovers M0B. Convert with mt.kappa_MT_from_forward_rate (which needs the
    # substrate's own S/V, so kappa_MT is geometry-dependent) and mt.dwell_time_from_fraction.
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
    'delta_chi_a_myelin': {
        'default': {
            'value': -0.1e-6,
            'unit': 'SI (dimensionless)',
            'field_T': None,   # susceptibility is field-independent
            'species': 'human',
            'method': 'GRE phase imaging + susceptibility tensor model',
            'source_key': 'liu2010',
            'location': 'Table 1 / Eq. 12: Δχ_a ≈ −0.1 ppm relative to water',
        },
        'alternatives': [
            {
                'value': -0.1e-6,
                'unit': 'SI (dimensionless)',
                'field_T': 3.0,
                'species': 'human',
                'method': 'GRE phase + hollow-cylinder model',
                'source_key': 'wharton2012',
                'location': 'Table 1: Δχ_a = −0.1 ppm used in model fits',
                'citation': {
                    'key': 'wharton2012',
                    'authors': 'Wharton S, Bowtell R',
                    'title': 'Fiber orientation-dependent white matter contrast in gradient '
                             'echo MRI',
                    'journal': 'Proceedings of the National Academy of Sciences',
                    'year': 2012,
                    'doi': '10.1073/pnas.1211075109',
                },
            },
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
        'citation': {
            'key': 'liu2010',
            'authors': 'Liu C',
            'title': 'Susceptibility tensor imaging',
            'journal': 'Magnetic Resonance in Medicine',
            'year': 2010,
            'doi': '10.1002/mrm.22391',
        },
        'description': (
            'Susceptibility anisotropy of myelinated white matter: difference '
            'between susceptibility parallel and perpendicular to the fibre axis, '
            'Δχ_a = χ_∥ − χ_⊥.  Negative because myelin is more diamagnetic along '
            'the axon than perpendicular to it (phospholipid bilayer geometry).  '
            'In SI units (dimensionless); 1 ppm = 1×10⁻⁶.  '
            'Enters the hollow-cylinder dipolar field as: '
            'ΔB_ea = (Δχ_a·B₀·sin²θ/2)·b²(1−g²)/r²·cos(2φ).'
        ),
        'note': (
            'Sign convention: negative Δχ_a means fibres parallel to B₀ are '
            'slightly less diamagnetic than fibres perpendicular to B₀.  '
            'The range −0.08 to −0.12 ppm spans most published estimates; '
            '−0.1 ppm is the community default.  '
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
    'kappa_membrane': {
        'default': {
            'value': 1e-5,
            'unit': 'm/s',
            'field_T': None,   # membrane permeability is geometric, field-independent
            'species': 'human',
            'method': 'filter-exchange imaging',
            'source_key': 'nilsson2013',
            'location': 'Section "Results", exchange rate ~10 s^-1 for WM; '
                        'converted to permeability kappa ~ 10^-5 m/s '
                        'using kappa = k * R / 3 for typical axon radius',
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
        'description': 'Axonal membrane permeability',
        'note': 'Derived from exchange rate measurements; depends on assumed axon geometry',
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
        'gamma_proton': get_value('gamma_proton'),
        'kappa': get_value('kappa_membrane', field_T, allow_nearest=True),
        # quantitative MT (two-pool). These are the MEASURED qMT observables, not the walk's
        # (kappa_MT, dwell_time) -- kappa_MT depends on the substrate's own S/V, so it cannot live in a
        # tissue table. Convert with mt.kappa_MT_from_forward_rate / mt.dwell_time_from_fraction, or
        # Substrate.with_mt(f_bound, k_forward, S_over_V).
        'mt_bound_pool_fraction': get_value('mt_bound_pool_fraction', field_T, allow_nearest=True),
        'mt_exchange_rate': get_value('mt_exchange_rate', field_T, allow_nearest=True),
        'T2_bound': get_value('T2_bound_pool', field_T, allow_nearest=True),
    }
