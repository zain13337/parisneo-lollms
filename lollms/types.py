from enum import Enum

class MSG_TYPE(Enum):
    # Messaging
    MSG_TYPE_CHUNK=0        # A chunk of a message (used for classical chat)
    MSG_TYPE_FULL=1         # A full message (for some personality the answer is sent in bulk)

    # Informations
    MSG_TYPE_EXCEPTION=2        # An exception occured
    MSG_TYPE_WARNING=3          # A warning occured
    MSG_TYPE_INFO=4             # An information to be shown to user

    # Steps
    MSG_TYPE_STEP=5             # An instant step (a step that doesn't need time to be executed)
    MSG_TYPE_STEP_START=6       # A step has started (the text contains an explanation of the step done by he personality)
    MSG_TYPE_STEP_PROGRESS=7    # The progress value (the text contains a percentage and can be parsed by the reception)
    MSG_TYPE_STEP_END=8         # A step has been done (the text contains an explanation of the step done by he personality)

    #Extra
    MSG_TYPE_JSON_INFOS=9       # A JSON output that is useful for summarizing the process of generation used by personalities like chain of thoughts and tree of thooughts
    MSG_TYPE_REF=10             # References (in form of  [text](path))
    MSG_TYPE_CODE=11            # A javascript code to execute
    MSG_TYPE_UI=12              # A vue.js component to show (we need to build some and parse the text to show it)
