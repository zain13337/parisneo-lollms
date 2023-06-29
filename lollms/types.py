from enum import Enum

class MSG_TYPE(Enum):
    # Messaging
    MSG_TYPE_CHUNK=0        # A chunk of a message (used for classical chat)
    MSG_TYPE_FULL=1         # A full message (for some personality the answer is sent in bulk)

    # Informations
    MSG_TYPE_INFO=3             # An information to be shown to user
    MSG_TYPE_EXCEPTION=2        # An exception occured

    # Steps
    MSG_TYPE_STEP=4             # An instant step (a step that doesn't need time to be executed)
    MSG_TYPE_STEP_START=4       # A step has started (the text contains an explanation of the step done by he personality)
    MSG_TYPE_STEP_PROGRESS=5    # The progress value (the text contains a percentage and can be parsed by the reception)
    MSG_TYPE_STEP_END=6         # A step has been done (the text contains an explanation of the step done by he personality)

    #Extra
    MSG_TYPE_REF=7          # References (in form of  [text](path))
    MSG_TYPE_CODE=8         # A javascript code to execute
    MSG_TYPE_UI=9           # A vue.js component to show (we need to build some and parse the text to show it)
