from fastapi import APIRouter
from lollms.server.elf_server import LOLLMSElfServer
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from lollms.types import MSG_TYPE
from lollms.utilities import detect_antiprompt, remove_text_from_string
from ascii_colors import ASCIIColors
class GenerateRequest(BaseModel):
    text: str
    n_predict: int = 1024
    stream: bool = False

router = APIRouter()
elf_server = LOLLMSElfServer.get_instance()

@router.post("/generate")
def generate(request_data: GenerateRequest):
    text = request_data.text
    n_predict = request_data.n_predict
    stream = request_data.stream
    
    if elf_server.binding is not None:
        if stream:
            output = {"text":""}
            def generate_chunks():
                def callback(chunk, chunk_type:MSG_TYPE=MSG_TYPE.MSG_TYPE_CHUNK):
                    # Yield each chunk of data
                    output["text"] += chunk
                    antiprompt = detect_antiprompt(output["text"])
                    if antiprompt:
                        ASCIIColors.warning(f"\nDetected hallucination with antiprompt: {antiprompt}")
                        output["text"] = remove_text_from_string(output["text"],antiprompt)
                        return False
                    else:
                        yield chunk
                        return True
                return iter(elf_server.binding.generate(text, n_predict, callback=callback))
            
            return StreamingResponse(generate_chunks())
        else:
            output = {"text":""}
            def callback(chunk, chunk_type:MSG_TYPE=MSG_TYPE.MSG_TYPE_CHUNK):
                # Yield each chunk of data
                output["text"] += chunk
                antiprompt = detect_antiprompt(output["text"])
                if antiprompt:
                    ASCIIColors.warning(f"\nDetected hallucination with antiprompt: {antiprompt}")
                    output["text"] = remove_text_from_string(output["text"],antiprompt)
                    return False
                else:
                    return True
            elf_server.binding.generate(text, n_predict, callback=callback)
            return output["text"]
    else:
        return None
