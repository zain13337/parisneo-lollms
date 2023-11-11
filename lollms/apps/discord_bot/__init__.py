import discord
from discord.ext import commands
from lollms.app import LollmsApplication
from lollms.paths import LollmsPaths
from lollms.main_config import LOLLMSConfig
import yaml
from pathlib import Path
from ascii_colors import ASCIIColors
from safe_store import TextVectorizer, GenericDataLoader, VisualizationMethod, VectorizationMethod
from typing import Tuple

context={
    "discussion":""
}

client = commands.Bot(command_prefix='!', intents=discord.Intents.all())

lollms_paths = LollmsPaths.find_paths(force_local=False, tool_prefix="lollms_discord_")
config = LOLLMSConfig.autoload(lollms_paths)
lollms_app = LollmsApplication("lollms_bot", config=config, lollms_paths=lollms_paths)

current_path = Path(__file__).resolve().parent
root_config_path = lollms_app.lollms_paths.personal_configuration_path / "discord_bot"
root_config_path.mkdir(parents=True, exist_ok=True)

root_data_path = lollms_app.lollms_paths.personal_data_path / "discord_bot"
root_data_path.mkdir(parents=True, exist_ok=True)

config_path = root_config_path / 'config.yaml'
text_vectorzer = TextVectorizer(
    database_path=root_data_path / "db.json",
    vectorization_method=VectorizationMethod.TFIDF_VECTORIZER,
    save_db=True
)
files = [f for f in root_data_path.iterdir() if f.suffix in [".txt", ".md", ".pdf"]]
for f in files:
    txt = GenericDataLoader.read_file(f)
    text_vectorzer.add_document(f, txt, 512, 128, False)
text_vectorzer.index()

if not config_path.exists():
    bot_token = input("Please enter your bot token: ")
    config = {'bot_token': bot_token, "summoning_word":"!lollms"}
    with open(config_path, 'w') as config_file:
        yaml.safe_dump(config, config_file)
else:
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    bot_token = config['bot_token']

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')
    print('------')

@client.event
async def on_member_join(member):
    channel = member.guild.system_channel
    if channel is not None:
        await channel.send(f'Welcome {member.mention} to the server! How can I assist you today?')

# def prepare_query(prompt, n_tokens: int = 0) -> Tuple[str, str, List[str]]:
#     """
#     Prepares the query for the model.

#     Args:
#         client_id (str): The client ID.
#         message_id (int): The message ID. Default is -1.
#         is_continue (bool): Whether the query is a continuation. Default is False.
#         n_tokens (int): The number of tokens. Default is 0.

#     Returns:
#         Tuple[str, str, List[str]]: The prepared query, original message content, and tokenized query.
#     """
    
#     # Define current message
#     current_message = messages[message_index]

#     # Build the conditionning text block
#     conditionning = self.personality.personality_conditioning

#     # Check if there are document files to add to the prompt
#     documentation = ""
#     if self.personality.persona_data_vectorizer:
#         if documentation=="":
#             documentation="!@>Documentation:\n"
#         docs, sorted_similarities = self.personality.persona_data_vectorizer.recover_text(current_message.content, top_k=self.config.data_vectorization_nb_chunks)
#         for doc, infos in zip(docs, sorted_similarities):
#             documentation += f"document chunk:\n{doc}"

    
#     if len(self.personality.text_files) > 0 and self.personality.vectorizer:
#         if documentation=="":
#             documentation="!@>Documentation:\n"
#         docs, sorted_similarities = self.personality.vectorizer.recover_text(current_message.content, top_k=self.config.data_vectorization_nb_chunks)
#         for doc, infos in zip(docs, sorted_similarities):
#             documentation += f"document chunk:\nchunk path: {infos[0]}\nchunk content:{doc}"

#     # Check if there is discussion history to add to the prompt
#     history = ""
#     if self.config.use_discussions_history and self.discussions_store is not None:
#         if history=="":
#             documentation="!@>History:\n"
#         docs, sorted_similarities = self.discussions_store.recover_text(current_message.content, top_k=self.config.data_vectorization_nb_chunks)
#         for doc, infos in zip(docs, sorted_similarities):
#             history += f"discussion chunk:\ndiscussion title: {infos[0]}\nchunk content:{doc}"

#     # Add information about the user
#     user_description=""
#     if self.config.use_user_name_in_discussions:
#         user_description="!@>User description:\n"+self.config.user_description


#     # Tokenize the conditionning text and calculate its number of tokens
#     tokens_conditionning = self.model.tokenize(conditionning)
#     n_cond_tk = len(tokens_conditionning)

#     # Tokenize the documentation text and calculate its number of tokens
#     if len(documentation)>0:
#         tokens_documentation = self.model.tokenize(documentation)
#         n_doc_tk = len(tokens_documentation)
#     else:
#         tokens_documentation = []
#         n_doc_tk = 0

#     # Tokenize the history text and calculate its number of tokens
#     if len(history)>0:
#         tokens_history = self.model.tokenize(history)
#         n_history_tk = len(tokens_history)
#     else:
#         tokens_history = []
#         n_history_tk = 0


#     # Tokenize user description
#     if len(user_description)>0:
#         tokens_user_description = self.model.tokenize(user_description)
#         n_user_description_tk = len(tokens_user_description)
#     else:
#         tokens_user_description = []
#         n_user_description_tk = 0


#     # Calculate the total number of tokens between conditionning, documentation, and history
#     total_tokens = n_cond_tk + n_doc_tk + n_history_tk + n_user_description_tk

#     # Calculate the available space for the messages
#     available_space = self.config.ctx_size - n_tokens - total_tokens

#     # Raise an error if the available space is 0 or less
#     if available_space<1:
#         raise Exception("Not enough space in context!!")

#     # Accumulate messages until the cumulative number of tokens exceeds available_space
#     tokens_accumulated = 0


#     # Initialize a list to store the full messages
#     full_message_list = []
#     # If this is not a continue request, we add the AI prompt
#     if not is_continue:
#         message_tokenized = self.model.tokenize(
#             "\n" +self.personality.ai_message_prefix.strip()
#         )
#         full_message_list.append(message_tokenized)
#         # Update the cumulative number of tokens
#         tokens_accumulated += len(message_tokenized)


#     # Accumulate messages starting from message_index
#     for i in range(message_index, -1, -1):
#         message = messages[i]

#         # Check if the message content is not empty and visible to the AI
#         if message.content != '' and (
#                 message.message_type <= MSG_TYPE.MSG_TYPE_FULL_INVISIBLE_TO_USER.value and message.message_type != MSG_TYPE.MSG_TYPE_FULL_INVISIBLE_TO_AI.value):

#             # Tokenize the message content
#             message_tokenized = self.model.tokenize(
#                 "\n" + self.config.discussion_prompt_separator + message.sender + ": " + message.content.strip())

#             # Check if adding the message will exceed the available space
#             if tokens_accumulated + len(message_tokenized) > available_space:
#                 break

#             # Add the tokenized message to the full_message_list
#             full_message_list.insert(0, message_tokenized)

#             # Update the cumulative number of tokens
#             tokens_accumulated += len(message_tokenized)

#     # Build the final discussion messages by detokenizing the full_message_list
#     discussion_messages = ""
#     for message_tokens in full_message_list:
#         discussion_messages += self.model.detokenize(message_tokens)

#     # Build the final prompt by concatenating the conditionning and discussion messages
#     prompt_data = conditionning + documentation + history + user_description + discussion_messages

#     # Tokenize the prompt data
#     tokens = self.model.tokenize(prompt_data)

#     # if this is a debug then show prompt construction details
#     if self.config["debug"]:
#         ASCIIColors.bold("CONDITIONNING")
#         ASCIIColors.yellow(conditionning)
#         ASCIIColors.bold("DOC")
#         ASCIIColors.yellow(documentation)
#         ASCIIColors.bold("HISTORY")
#         ASCIIColors.yellow(history)
#         ASCIIColors.bold("DISCUSSION")
#         ASCIIColors.hilight(discussion_messages,"!@>",ASCIIColors.color_yellow,ASCIIColors.color_bright_red,False)
#         ASCIIColors.bold("Final prompt")
#         ASCIIColors.hilight(prompt_data,"!@>",ASCIIColors.color_yellow,ASCIIColors.color_bright_red,False)
#         ASCIIColors.info(f"prompt size:{len(tokens)} tokens") 
#         ASCIIColors.info(f"available space after doc and history:{available_space} tokens") 

#     # Return the prepared query, original message content, and tokenized query
#     return prompt_data, current_message.content, tokens

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.content.startswith(config["summoning_word"]):
        prompt = message.content[len(config["summoning_word"])+1:]
        context['discussion']+= "\n!@>" + message.author.name +": "+  prompt + "\n" + f"{lollms_app.personality.ai_message_prefix}"
        context['current_response']=""
        print("Chatting")
        try:
            docs, _ = text_vectorzer.recover_text(prompt,3)
            docs = "Use the content of those documentation chunks to enhance your answers\n!@>Documentation:\n"+'\n'.join(docs)
        except:
            docs=""
        context_text = f"""{lollms_app.personality.personality_conditioning}
!@>informations:
Current model:{lollms_app.config.model_name}
Current personality:{lollms_app.personality.name}
{docs}
"""+"{{discussion}}"
        def callback(text, type=None):
            antiprompt = lollms_app.personality.detect_antiprompt(context['current_response'])
            if antiprompt:
                ASCIIColors.warning(f"\nDetected hallucination with antiprompt: {antiprompt}")
                context['current_response'] = lollms_app.remove_text_from_string(context['current_response'],antiprompt)
                return False

            context['current_response'] += text
            print(text,end="")
            return True
        
        ASCIIColors.green("Warming up ...")
        lollms_app.safe_generate(context_text, n_predict=1024, callback=callback, placeholder={"discussion":context['discussion']},place_holders_to_sacrifice=["discussion"], debug=True)

        print()
        context['discussion'] += context['current_response'][0:2000]
        await message.channel.send(context['current_response'][0:2000])
    elif message.content.startswith('!mount'):
        personality_name =  message.content[len('!mount')+1:]
        lollms_app.config.personalities.append(personality_name)
        lollms_app.personality = lollms_app.mount_personality(len(lollms_app.config.personalities)-1)
        lollms_app.config.active_personality_id = len(lollms_app.config.personalities)-1
        
        await message.channel.send(f"Personality {personality_name} mounted successfuly")
    elif message.content.startswith('!list'):
        await message.channel.send(f"Mounted personalities:\n{[p.name for p in lollms_app.mounted_personalities]}")
    elif message.content.startswith('!select'):
        personality_name =  message.content[len('!select')+1:]
        index = 0
        for i in range(len(lollms_app.mounted_personalities)):
            if lollms_app.mounted_personalities[i].name.lower().strip()==personality_name.lower():
                index = i
        lollms_app.config.active_personality_id = index
        await message.channel.send(f"Activated personality:\n{personality_name}")
    elif message.content.startswith('!reset'):
        context["discussion"]=""
        await message.channel.send(f"Content reset")

    elif message.content.startswith('!install'):
        response = "To install lollms, make sure you have installed the currently supported python version (consult the repository to verify what version is currently supported, but as of 10/22/2023, the version is 3.10).\nThen you can follow these steps:\n1. Open your command line interface.\n2. Navigate to the directory where you want to install lollms.\n3. Run the following command to clone the lollms repository: `git clone https://github.com/lollms/lollms.git`.\n4. Once the repository is cloned, navigate into the lollms directory.\n5. Run `pip install -r requirements.txt` to install the required dependencies.\n6. You can now use lollms by importing it in your Python code."
        await message.channel.send(response)

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')
    print('------')

@client.event
async def on_member_join(member):
    channel = member.guild.system_channel
    if channel is not None:
        await channel.send(f'Welcome {member.mention} to the server! How can I assist you today?')

client.run(bot_token)
