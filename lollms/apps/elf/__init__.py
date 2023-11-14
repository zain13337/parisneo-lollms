from lollms.app import LollmsApplication
from lollms.paths import LollmsPaths
from lollms.main_config import LOLLMSConfig
import sys
import time
maxtry=10
from pathlib import Path
import json
import re
import random
from flask import Flask, make_response, request, abort
from flask.json import jsonify
from typing import Callable
import string
import argparse
from ascii_colors import ASCIIColors
BUNDLES=4
MAXWORDS=1048
DEBUG=True
class Elf(LollmsApplication):
    def __init__(self):
        pass
    def init(self, custom_default_cfg_path):
        lollms_paths = LollmsPaths.find_paths(custom_global_paths_cfg_path=custom_default_cfg_path, tool_prefix="lollms_elf_")
        config = LOLLMSConfig.autoload(lollms_paths, None)
        super().__init__("Elf", config, lollms_paths)

    def split_fibers(self,fibers, max_words=MAXWORDS):
        # Split each fiber into chunks of up to max_words words
        sfibers = []
        for fiber in fibers:
            words = fiber.split()
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i : i + max_words])
                sfibers.append(chunk)
        return sfibers

    def refactor_into_fiber_bundles(self, lines, bundle_size):
        bundles = []
        temp = []
        for line in lines:
            # Split the line into fibers
            # fibers = line.split('.')
            fibers = re.split(r"[\.\n]", line)

            # Filter out empty lines or lines with only whitespace
            fibers = [fiber.strip() for fiber in fibers if re.search(r"\S", fiber)]

            # Add filtered fibers to the current bundle
            temp.extend(self.split_fibers(fibers))
        # now lete
        current_bundle = []
        # print(temp)
        for line in temp:
            current_bundle.append(line)

            # Check if the current bundle size exceeds the desired bundle size
            if len(current_bundle) >= bundle_size:
                # Add the current bundle to the list of bundles
                bundles.append(current_bundle)
                # Start a new bundle
                current_bundle = []

        # Add the last bundle if it's not empty
        if current_bundle:
            bundles.append(current_bundle)

        return bundles


    def read_input_file(self, file_path):
        with open(file_path, "r") as file:
            lines = file.readlines()

            lines = self.refactor_into_fiber_bundles(lines, BUNDLES)

            with open("debug.txt", "w") as fo:
                for line in lines:
                    fo.write("|\n".join(line))
        for line in lines:
            self.text_ring_buffer.append(
                self.personality.user_message_prefix + "\n".join(line)
            )
        print("start COUNT",len(self.text_ring_buffer))

    def safe_generate(
                        self, 
                        full_discussion:str, 
                        n_predict=None, 
                        callback: Callable[[str, int, dict], bool]=None,
                        temperature=0.1,
                        top_k=50,
                        top_p=0.9,
                        repeat_penalty=1.3,
                        last_n_tokens=60,
                        seed=-1,
                        n_threads=8,
                        batch_size=1,
                        stream=None):
        """safe_generate

        Args:
            full_discussion (string): A prompt or a long discussion to use for generation
            callback (_type_, optional): A callback to call for each received token. Defaults to None.

        Returns:
            str: Model output
        """
        if n_predict == None:
            n_predict =self.personality.model_n_predicts
        tk = self.personality.model.tokenize(full_discussion)
        n_tokens = len(tk)
        fd = self.personality.model.detokenize(tk[-min(self.config.ctx_size-self.n_cond_tk,n_tokens):])
        self.bot_says = ""
        output = self.personality.model.generate(
                        self.personality.personality_conditioning+fd, 
                        n_predict=n_predict, 
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repeat_penalty=repeat_penalty,
                        last_n_tokens=last_n_tokens,
                        seed=seed,
                        n_threads=n_threads,
                        batch_size=batch_size,                       
                        callback=callback)
        return output

    def gen_rewrite(self):
        topic = "Snippet"
        target= "Protobuf Server"
        return random.choice(
            [
                f"Transform this {topic} into a Python code representation of a {target}.",
                f"Generate a Python code snippet for a {target} that implements this {topic}.",
                f"Craft a Python implementation of a {target} that embodies the essence of this {topic}.",
            ]
        )


    def callback(self, text, type=None, metadata: dict = {}):
        if DEBUG:
            print("DBG:" + text, end="")
            sys.stdout.flush()
        return True

# set up the Flask application
app = Flask(__name__)


cv = Elf()
    # input_file_path = "user_input.txt"
    # try:
    #     cv.read_input_file(input_file_path)

    #     cv.start_conversation2()
    # except Exception as e:
    #     print(e)
    # raise e
models = {}


@app.route("/v1/models", methods=['GET'])
def models():
    data = [
        {
            "id": model,
            "object": "model",
            "owned_by": "",
            "tokens": 99999,
            "fallbacks": None,
            "endpoints": [
                "/v1/chat/completions"
            ],
            "limits": None,
            "permission": []
        }
        for model in cv.binding.list_models(cv.config)
    ]
    return {'data': data, 'object': 'list'}
     
@app.route("/chat/completions", methods=['POST'])
@app.route("/v1/chat/completions", methods=['POST'])
@app.route("/", methods=['POST'])
def chat_completions():
    request_data = request.get_json()
    model = request_data.get('model', None).replace("neuro-", "")
    messages = request_data.get('messages')
    stream = request_data.get('stream', False)
    streaming_ = request_data.get('stream', False)
    temperature = request_data.get('temperature', 1.0)
    top_p = request_data.get('top_p', 1.0)
    max_tokens = request_data.get('max_tokens', 1024)

    if model is not None:
        # TODO add model selection
        pass

    full_discussion = ""
    for message in messages:
        full_discussion += f'{message["role"]}: {message["content"]}\n'
    response = cv.safe_generate(full_discussion=full_discussion, temperature=temperature, top_p=top_p, n_predict=max_tokens)

    completion_id = "".join(random.choices(string.ascii_letters + string.digits, k=28))
    completion_timestamp = int(time.time())
    
    if not streaming_:
        completion_timestamp = int(time.time())
        completion_id = ''.join(random.choices(
            'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789', k=28))

        return {
            "id": f"chatcmpl-{completion_id}",
            "object": "chat.completion",
            "created": completion_timestamp,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            },
        }

    def stream():
        nonlocal response
        for token in response:
            completion_timestamp = int(time.time())
            completion_id = ''.join(random.choices(
                'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789', k=28))

            completion_data = {
                'id': f'chatcmpl-{completion_id}',
                'object': 'chat.completion.chunk',
                'created': completion_timestamp,
                'choices': [
                    {
                        'delta': {
                            'content': token
                        },
                        'index': 0,
                        'finish_reason': None
                    }
                ]
            }
            #print(token)
            #print(completion_data)
            #print('data: %s\n\n' % json.dumps(completion_data, separators=(',' ':')))
            yield 'data: %s\n\n' % json.dumps(completion_data, separators=(',' ':'))
            time.sleep(0.02)
    print('===Start Streaming===')
    return app.response_class(stream(), mimetype='text/event-stream')

# define the engines endpoint    
@app.route('/v1/engines')
@app.route('/v1/models')
@app.route("/models", methods=['GET'])
def v1_engines():
    return make_response(jsonify({
        'data': [{
            'object': 'engine',
            'id': id,
            'ready': True,
            'owner': 'huggingface',
            'permissions': None,
            'created': None
        } for id in models.keys()]
    }))




@app.route('/v1/embeddings', methods=['POST'])
@app.route('/embeddings', methods=['POST'])
def create_embedding():
    j_input = request.get_json()
    #model = embedding_processing()
    embedding = cv.model.embedding(text_list=j_input['input'])
    return jsonify(
        embedding
        )

    
@app.route("/v1/dashboard/billing/subscription", methods=['GET'])
@app.route("/dashboard/billing/subscription", methods=['GET'])
def billing_subscription():
    return jsonify({
  "object": "billing_subscription",
  "has_payment_method": True,
  "canceled": False,
  "canceled_at": None,
  "delinquent": None,
  "access_until": 2556028800,
  "soft_limit": 6944500,
  "hard_limit": 166666666,
  "system_hard_limit": 166666666,
  "soft_limit_usd": 416.67,
  "hard_limit_usd": 9999.99996,
  "system_hard_limit_usd": 9999.99996,
  "plan": {
    "title": "Pay-as-you-go",
    "id": "payg"
  },
  "primary": True,
  "account_name": "OpenAI",
  "po_number": None,
  "billing_email": None,
  "tax_ids": None,
  "billing_address": {
    "city": "New York",
    "line1": "OpenAI",
    "country": "US",
    "postal_code": "NY10031"
  },
  "business_address": None
}
)


@app.route("/v1/dashboard/billing/usage", methods=['GET'])
@app.route("/dashboard/billing/usage", methods=['GET'])
def billing_usage():
    return jsonify({
  "object": "list",
  "daily_costs": [
    {
      "timestamp": time.time(),
      "line_items": [
        {
          "name": "GPT-4",
          "cost": 0.0
        },
        {
          "name": "Chat models",
          "cost": 1.01
        },
        {
          "name": "InstructGPT",
          "cost": 0.0
        },
        {
          "name": "Fine-tuning models",
          "cost": 0.0
        },
        {
          "name": "Embedding models",
          "cost": 0.0
        },
        {
          "name": "Image models",
          "cost": 16.0
        },
        {
          "name": "Audio models",
          "cost": 0.0
        }
      ]
    }
  ],
  "total_usage": 1.01
}
)


@app.route("/v1/providers", methods=['GET'])
@app.route("/providers", methods=['GET'])
def providers():
  #todo : implement
  return jsonify({})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', '-hst', default=None, help='Host name')
    parser.add_argument('--port', '-prt', default=None, help='Port number')

    
    parser.add_argument('--reset_personal_path', action='store_true', help='Reset the personal path')
    parser.add_argument('--reset_config', action='store_true', help='Reset the configurations')
    parser.add_argument('--reset_installs', action='store_true', help='Reset all installation status')
    parser.add_argument('--default_cfg_path', type=str, default=None, help='Reset all installation status')

    ASCIIColors.yellow("         _             _            _             _            _   _         _        ")
    ASCIIColors.yellow("        _\ \          /\ \         _\ \          _\ \         /\_\/\_\ _    / /\      ")
    ASCIIColors.yellow("       /\__ \        /  \ \       /\__ \        /\__ \       / / / / //\_\ / /  \     ")
    ASCIIColors.yellow("      / /_ \_\      / /\ \ \     / /_ \_\      / /_ \_\     /\ \/ \ \/ / // / /\ \__  ")
    ASCIIColors.yellow("     / / /\/_/     / / /\ \ \   / / /\/_/     / / /\/_/    /  \____\__/ // / /\ \___\ ")
    ASCIIColors.yellow("    / / /         / / /  \ \_\ / / /         / / /        / /\/________/ \ \ \ \/___/ ")
    ASCIIColors.yellow("   / / /         / / /   / / // / /         / / /        / / /\/_// / /   \ \ \       ")
    ASCIIColors.yellow("  / / / ____    / / /   / / // / / ____    / / / ____   / / /    / / /_    \ \ \      ")
    ASCIIColors.yellow(" / /_/_/ ___/\ / / /___/ / // /_/_/ ___/\ / /_/_/ ___/\/ / /    / / //_/\__/ / /      ")
    ASCIIColors.yellow("/_______/\__\// / /____\/ //_______/\__\//_______/\__\/\/_/    / / / \ \/___/ /       ")
    ASCIIColors.yellow("\_______\/    \/_________/ \_______\/    \_______\/            \/_/   \_____\/        ")
    ASCIIColors.yellow("         _            _             _                                                 ")
    ASCIIColors.yellow("        /\ \         _\ \          /\ \                                               ")
    ASCIIColors.yellow("       /  \ \       /\__ \        /  \ \                                              ")
    ASCIIColors.yellow("      / /\ \ \     / /_ \_\      / /\ \ \                                             ")
    ASCIIColors.yellow("     / / /\ \_\   / / /\/_/     / / /\ \_\                                            ")
    ASCIIColors.yellow("    / /_/_ \/_/  / / /         / /_/_ \/_/                                            ")
    ASCIIColors.yellow("   / /____/\    / / /         / /____/\                                               ")
    ASCIIColors.yellow("  / /\____\/   / / / ____    / /\____\/                                               ")
    ASCIIColors.yellow(" / / /______  / /_/_/ ___/\ / / /                                                     ")
    ASCIIColors.yellow("/ / /_______\/_______/\__\// / /                                                      ")
    ASCIIColors.yellow("\/__________/\_______\/    \/_/                                                       ")
                                                                                      

    args = parser.parse_args()

    if args.reset_personal_path:
        LollmsPaths.reset_configs()

    cv.init(args.default_cfg_path)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()