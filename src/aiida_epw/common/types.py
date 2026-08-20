import enum


class RestartType(enum.Enum):
    """Enumeration of EPW run/restart modes."""

    NONE = "none"
    EPHWRITE = "ephwrite"
    EPHREAD = "ephread"
    EPHWRITE_RESTART = "ephwrite_restart"
    EPHREAD_RESTART = "ephread_restart"
    EPWREAD = "epwread"


RESTART_TYPE_DEFAULTS = {
    RestartType.NONE: {
        "epwread": False,
        "epwwrite": True,
        "restart": False,
        "ep_coupling": True,
        "elph": True,
        "epbwrite": True,
        "epbread": False,
    },
    RestartType.EPHWRITE: {
        "epwread": True,
        "epwwrite": False,
        "restart": False,
        "ep_coupling": True,
        "elph": True,
        "ephwrite": True,
    },
    RestartType.EPHWRITE_RESTART: {
        "epwread": True,
        "epwwrite": False,
        "restart": True,
        "ep_coupling": True,
        "elph": True,
        "ephwrite": True,
    },
    RestartType.EPHREAD: {
        "epwread": True,
        "restart": False,
        "ep_coupling": False,
        "elph": False,
        "ephwrite": False,
    },
    RestartType.EPHREAD_RESTART: {
        "epwread": True,
        "restart": True,
        "ep_coupling": False,
        "elph": False,
        "ephwrite": False,
    },
    RestartType.EPWREAD: {
        "epwread": True,
        "epwwrite": False,
        "epbwrite": False,
        "epbread": False,
        "ep_coupling": True,
        "elph": True,
        "restart": False,
    },
}
