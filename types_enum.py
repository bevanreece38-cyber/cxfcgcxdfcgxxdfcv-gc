from enum import Enum


class SafetyStatus(Enum):
    OK      = 0
    WARNING = 1
    LAND    = 2


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