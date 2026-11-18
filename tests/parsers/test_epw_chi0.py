"""Tests for parsing the Wannier-basis susceptibility output headers."""

from aiida_epw.parsers.epw import EpwParser

CONTENT = """# EPW chi0_GG with Wannier density form factors: PER SPIN, positive static, eV^-1 per cell
# kind: 0 static, 1 imaginary axis (w = i nu), 2 real axis (w + i eta)
# iq     1  q_crys   0.1250000   0.1250000   0.1250000  nG  103  stoner_eig  0.652728  N0[/eV/spin]  0.939345  nesting  1.84163E+02
  0      0.000000   8.01111346E-01   6.10234829E-20
# iq     2  q_crys   0.1250000   0.1250000   0.3750000  nG   98  stoner_eig  0.721900  N0[/eV/spin]  0.939345  nesting  1.55745E+02
  0      0.000000   7.5E-01   0.0
"""


def test_parse_chi0_qpoint_headers():
    qpoints = EpwParser.parse_chi0_qpoint_headers(CONTENT)

    assert [q["iq"] for q in qpoints] == [1, 2]
    assert qpoints[0]["q_crystal"] == [0.125, 0.125, 0.125]
    assert qpoints[0]["number_of_g_vectors"] == 103
    assert qpoints[0]["stoner_eigenvalue"] == 0.652728
    assert qpoints[1]["dos_ef_per_spin"] == 0.939345
    assert qpoints[1]["nesting"] == 155.745


def test_parse_chi0_qpoint_headers_empty():
    assert EpwParser.parse_chi0_qpoint_headers(None) == []
    assert EpwParser.parse_chi0_qpoint_headers("# no q headers\n 0 0.0 1.0\n") == []
