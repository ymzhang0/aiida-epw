# -*- coding: utf-8 -*-
"""Data types for EPW plugin."""

import enum


class CalculationTypes(enum.Enum):
    """Enumeration of EPW calculation types."""

    WANNIERIZE = "wannierize"
    ELIASHBERG = "eliashberg"
    TRANSPORT = "transport"
    POLARON = "polaron"


class RestartType(enum.Enum):
    """Enumeration of EPW run/restart modes."""

    EPHWRITE = "ephwrite"
    EPHREAD = "ephread"
    EPHWRITE_RESTART = "ephwrite_restart"
