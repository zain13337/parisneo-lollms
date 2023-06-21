from enum import Enum

class MSG_TYPE(Enum):
    MSG_TYPE_CHUNK=0
    MSG_TYPE_FULL=1
    MSG_TYPE_META=2
    MSG_TYPE_REF=3
    MSG_TYPE_CODE=4
    MSG_TYPE_UI=5
