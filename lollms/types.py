from enum import Enum

class MSG_TYPE(Enum):
    MSG_TYPE_CHUNK=0        # A chunk of a message (used for classical chat)
    MSG_TYPE_FULL=1         # A full message (for some personality the answer is sent in bulk)
    MSG_TYPE_EXCEPTION=2    # An exception occured
    MSG_TYPE_STEP=3         # A step has been done (the text contains an explanation of the step done by he personality)
    MSG_TYPE_PROGRESS=4     # The progress value (the text contains a percentage and can be parsed by the reception)
    MSG_TYPE_META=5         # Generic metadata
    MSG_TYPE_REF=6          # References (in form of  []() )
    MSG_TYPE_CODE=7         # A HTML code to be embedded in the answer
    MSG_TYPE_UI=8           # A vue.js component to show
