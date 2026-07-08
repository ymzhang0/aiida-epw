# -*- coding: utf-8 -*-
"""Data types for EPW plugin."""

import enum


class CalculationTypes(enum.Enum):
    """Enumeration of EPW calculation types."""

    WANNIERIZE = "wannierize"
    ELIASHBERG = "eliashberg"
    TRANSPORT = "transport"
    POLARON = "polaron"
    BANDS = "bands"


class RestartType(enum.Enum):
    """Enumeration of EPW run/restart modes."""

    NONE = "none"
    EPHWRITE = "ephwrite"
    EPHREAD = "ephread"
    EPHWRITE_RESTART = "ephwrite_restart"
    EPWREAD = "epwread"
