"""
project: lollms
file: lollms_generator.py 
author: ParisNeo
description: 
    This module contains a set of FastAPI routes that provide information about the Lord of Large Language and Multimodal Systems (LoLLMs) Web UI
    application. These routes are specific to the generation process

"""

from fastapi import APIRouter, Request
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
    temperature: float = 0.4
    top_k: int = 50
    top_p: float = 0.6
    repeat_penalty: float = 1.3
    repeat_last_n: int = 40
    seed: int = -1
    n_threads: int = 1

class V1ChatGenerateRequest(BaseModel):
    """
    Data model for the V1 Chat Generate Request.

    Attributes:
    - model: str representing the model to be used for text generation.
    - messages: list of messages to be used as prompts for text generation.
    - stream: bool indicating whether to stream the generated text or not.
    - temperature: float representing the temperature parameter for text generation.
    - max_tokens: float representing the maximum number of tokens to generate.
    """    
    model: str
    messages: list
    stream: bool
    temperature: float
    max_tokens: float


class V1InstructGenerateRequest(BaseModel):
    """
    Data model for the V1 Chat Generate Request.

    Attributes:
    - model: str representing the model to be used for text generation.
    - messages: list of messages to be used as prompts for text generation.
    - stream: bool indicating whether to stream the generated text or not.
    - temperature: float representing the temperature parameter for text generation.
    - max_tokens: float representing the maximum number of tokens to generate.
    """    
    model: str
    prompt: str
    stream: bool
    temperature: float
    max_tokens: float

# ----------------------- Defining router and main class ------------------------------

router = APIRouter()
elf_server = LOLLMSElfServer.get_instance()


# ----------------------------------- Generation status -----------------------------------------

@router.get("/get_generation_status")
def get_generation_status():
    return {"status":elf_server.busy}


# ----------------------------------- Generation -----------------------------------------
@router.post("/generate")
def lollms_generate(request_data: Request):
    """
    Endpoint for generating text from prompts using the lollms fastapi server.

    Args:
    Data model for the Generate Request.

    Attributes:
    - text: str representing the input text prompt for text generation.
    - n_predict: int representing the number of predictions to generate.
    - stream: bool indicating whether to stream the generated text or not.
    - temperature: float representing the temperature parameter for text generation.
    - top_k: int representing the top_k parameter for text generation.
    - top_p: float representing the top_p parameter for text generation.
    - repeat_penalty: float representing the repeat_penalty parameter for text generation.
    - repeat_last_n: int representing the repeat_last_n parameter for text generation.
    - seed: int representing the seed for text generation.
    - n_threads: int representing the number of threads for text generation.

    
    Returns:
    - If the elf_server binding is not None:
        - If stream is True, returns a StreamingResponse of generated text chunks.
        - If stream is False, returns the generated text as a string.
    - If the elf_server binding is None, returns None.
    """ 
    text = request_data["text"]
    n_predict = request_data.get("n_predict", 1024)
    stream = request_data.get("stream", False)
    
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
                return iter(elf_server.binding.generate(
                                            text, 
                                            n_predict, 
                                            callback=callback, 
                                            temperature=request_data.temperature,
                                            top_k=request_data.top_k, 
                                            top_p=request_data.top_p,
                                            repeat_penalty=request_data.repeat_penalty,
                                            repeat_last_n=request_data.repeat_last_n,
                                            seed=request_data.seed,
                                            n_threads=request_data.n_threads
                                        ))
            
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
            elf_server.binding.generate(
                                            text, 
                                            n_predict, 
                                            callback=callback,
                                            temperature=request_data.temperature,
                                            top_k=request_data.top_k, 
                                            top_p=request_data.top_p,
                                            repeat_penalty=request_data.repeat_penalty,
                                            repeat_last_n=request_data.repeat_last_n,
                                            seed=request_data.seed,
                                            n_threads=request_data.n_threads
                                        )
            return output["text"]
    else:
        return None
    

# openai compatible generation
@router.post("/v1/chat/completions")
def v1_chat_generate(request_data: V1ChatGenerateRequest):
    """
    Endpoint for generating text from prompts using the lollms fastapi server in chat completion mode.
    This endpoint is compatible with open ai API and mistralAI API
    Args:
    - request_data: GenerateRequest object containing the input text, number of predictions, and stream flag.

    Returns:
    - If the elf_server binding is not None:
        - If stream is True, returns a StreamingResponse of generated text chunks.
        - If stream is False, returns the generated text as a string.
    - If the elf_server binding is None, returns None.
    """    
    messages = request_data.messages
    text = ""
    for message in messages:
        text += f"{message['role']}: {message['content']}\n"
    n_predict = request_data.max_tokens
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
                return iter(elf_server.binding.generate(
                                            text, 
                                            n_predict, 
                                            callback=callback, 
                                            temperature=request_data.temperature
                                        ))
            
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
            elf_server.binding.generate(
                                            text, 
                                            n_predict, 
                                            callback=callback,
                                            temperature=request_data.temperature
                                        )
            return output["text"]
    else:
        return None




# openai compatible generation
@router.post("/v1/completions")
def v1_instruct_generate(request_data: V1InstructGenerateRequest):
    """
    Endpoint for generating text from prompts using the lollms fastapi server in instruct completion mode.
    This endpoint is compatible with open ai API and mistralAI API
    Args:
    - request_data: GenerateRequest object containing the input text, number of predictions, and stream flag.

    Returns:
    - If the elf_server binding is not None:
        - If stream is True, returns a StreamingResponse of generated text chunks.
        - If stream is False, returns the generated text as a string.
    - If the elf_server binding is None, returns None.
    """    
   
    text = request_data.prompt
    n_predict = request_data.max_tokens
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
                return iter(elf_server.binding.generate(
                                            text, 
                                            n_predict, 
                                            callback=callback, 
                                            temperature=request_data.temperature
                                        ))
            
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
            elf_server.binding.generate(
                                            text, 
                                            n_predict, 
                                            callback=callback,
                                            temperature=request_data.temperature
                                        )
            return output["text"]
    else:
        return None



@router.post("/stop_gen")
def stop_gen():
    elf_server.cancel_gen = True
    return {"status": True} 
