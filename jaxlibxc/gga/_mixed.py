"""Mixed/hybrid GGA functionals.

B3LYP, PBE0, B3PW91, and other hybrid GGA exchange-correlation functionals.
"""

from .._mixed import register_mixed


# B3LYP (ID 402)
# E = (1 - a0 - ax)*LDA_X + ax*B88_X + (1 - ac)*VWN_RPA + ac*LYP + a0*HF
# a0=0.20, ax=0.72, ac=0.81
register_mixed(
    number=402,
    name='hyb_gga_xc_b3lyp',
    component_specs=[
        (1.0 - 0.20 - 0.72, 'lda_x'),       # LDA exchange
        (0.72, 'gga_x_b88'),                  # B88 GGA exchange
        (1.0 - 0.81, 'lda_c_vwn_rpa'),        # VWN RPA correlation
        (0.81, 'gga_c_lyp'),                  # LYP GGA correlation
    ],
    hyb_exx=0.20,
)

# B3LYP5 (ID 475) - uses VWN5 instead of VWN_RPA
register_mixed(
    number=475,
    name='hyb_gga_xc_b3lyp5',
    component_specs=[
        (1.0 - 0.20 - 0.72, 'lda_x'),
        (0.72, 'gga_x_b88'),
        (1.0 - 0.81, 'lda_c_vwn'),           # VWN5
        (0.81, 'gga_c_lyp'),
    ],
    hyb_exx=0.20,
)

# B3PW91 (ID 401)
register_mixed(
    number=401,
    name='hyb_gga_xc_b3pw91',
    component_specs=[
        (1.0 - 0.20 - 0.72, 'lda_x'),
        (0.72, 'gga_x_b88'),
        (1.0 - 0.81, 'lda_c_pw'),
        (0.81, 'gga_c_pw91'),
    ],
    hyb_exx=0.20,
)

# PBE0 / PBE1PBE (ID 406)
# E = 0.75*PBE_X + PBE_C + 0.25*HF
register_mixed(
    number=406,
    name='hyb_gga_xc_pbeh',
    component_specs=[
        (0.75, 'gga_x_pbe'),
        (1.0, 'gga_c_pbe'),
    ],
    hyb_exx=0.25,
)
