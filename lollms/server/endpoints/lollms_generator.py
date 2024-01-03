from fastapi import APIRouter
from lollms.server.elf_server import LOLLMSElfServer
from pydantic import BaseModel
from starlette.responses import StreamingResponse

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
            def generate_chunks():
                def callback(chunk):
                    # Yield each chunk of data
                    yield chunk
                
                elf_server.binding.generate(text, n_predict, callback=callback)
            
            return StreamingResponse(generate_chunks())
        else:
            output = elf_server.binding.generate(text, n_predict)
            return output
    else:
        return None
