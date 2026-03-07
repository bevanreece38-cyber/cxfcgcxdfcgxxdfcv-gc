from enum import Enum


class SafetyStatus(Enum):
    OK      = "OK"
    WARNING = "WARNING"
    HOVER   = "HOVER"
    LAND    = "LAND"


class TrackerState(Enum):
    IDLE        = "IDLE"
    ACQUIRING   = "ACQUIRING"
    TRACKING    = "TRACKING"
    DEAD_RECKON = "DEAD_RECKON"
    LOST        = "LOST"
    STRIKING    = "STRIKING"


class ControlState(Enum):
    MANUAL   = "MANUAL"
    COMPUTER = "COMPUTER"