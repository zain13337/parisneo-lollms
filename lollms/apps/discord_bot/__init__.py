import discord
from discord.ext import commands
from lollms.app import LollmsApplication
from lollms.paths import LollmsPaths
from lollms.main_config import LOLLMSConfig
import yaml
from pathlib import Path
from ascii_colors import ASCIIColors
from safe_store import TextVectorizer, GenericDataLoader, VisualizationMethod, VectorizationMethod
from typing import Tuple, List

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

def prepare_query(discussion, prompt, n_tokens: int = 0) -> Tuple[str, str, List[str]]:
    """
    Prepares the query for the model.

    Args:
        client_id (str): The client ID.
        message_id (int): The message ID. Default is -1.
        is_continue (bool): Whether the query is a continuation. Default is False.
        n_tokens (int): The number of tokens. Default is 0.

    Returns:
        Tuple[str, str, List[str]]: The prepared query, original message content, and tokenized query.
    """

    # Define current message
    current_message = prompt

    # Build the conditionning text block
    conditionning = lollms_app.personality.personality_conditioning

    # Check if there are document files to add to the prompt
    documentation = ""
    if lollms_app.personality.persona_data_vectorizer:
        if documentation=="":
            documentation="!@>Documentation:\n"
        docs, sorted_similarities = lollms_app.personality.persona_data_vectorizer.recover_text(current_message, top_k=lollms_app.config.data_vectorization_nb_chunks)
        for doc, infos in zip(docs, sorted_similarities):
            documentation += f"document chunk:\n{doc}"


    if len(lollms_app.personality.text_files) > 0 and lollms_app.personality.vectorizer:
        if documentation=="":
            documentation="!@>Documentation:\n"
        docs, sorted_similarities = lollms_app.personality.vectorizer.recover_text(current_message.content, top_k=lollms_app.config.data_vectorization_nb_chunks)
        for doc, infos in zip(docs, sorted_similarities):
            documentation += f"document chunk:\nchunk path: {infos[0]}\nchunk content:{doc}"

    # Check if there is discussion history to add to the prompt
    history = ""
    if lollms_app.config.use_discussions_history and lollms_app.discussions_store is not None:
        if history=="":
            documentation="!@>History:\n"
        docs, sorted_similarities = lollms_app.discussions_store.recover_text(current_message.content, top_k=lollms_app.config.data_vectorization_nb_chunks)
        for doc, infos in zip(docs, sorted_similarities):
            history += f"discussion chunk:\ndiscussion title: {infos[0]}\nchunk content:{doc}"

    # Add information about the user
    user_description=""
    if lollms_app.config.use_user_name_in_discussions:
        user_description="!@>User description:\n"+lollms_app.config.user_description


    # Tokenize the conditionning text and calculate its number of tokens
    tokens_conditionning = lollms_app.model.tokenize(conditionning)
    n_cond_tk = len(tokens_conditionning)

    # Tokenize the documentation text and calculate its number of tokens
    if len(documentation)>0:
        tokens_documentation = lollms_app.model.tokenize(documentation)
        n_doc_tk = len(tokens_documentation)
    else:
        tokens_documentation = []
        n_doc_tk = 0

    # Tokenize the history text and calculate its number of tokens
    if len(history)>0:
        tokens_history = lollms_app.model.tokenize(history)
        n_history_tk = len(tokens_history)
    else:
        tokens_history = []
        n_history_tk = 0


    # Tokenize user description
    if len(user_description)>0:
        tokens_user_description = lollms_app.model.tokenize(user_description)
        n_user_description_tk = len(tokens_user_description)
    else:
        tokens_user_description = []
        n_user_description_tk = 0


    # Calculate the total number of tokens between conditionning, documentation, and history
    total_tokens = n_cond_tk + n_doc_tk + n_history_tk + n_user_description_tk

    # Calculate the available space for the messages
    available_space = lollms_app.config.ctx_size - n_tokens - total_tokens

    # Raise an error if the available space is 0 or less
    if available_space<1:
        raise Exception("Not enough space in context!!")

    # Accumulate messages until the cumulative number of tokens exceeds available_space
    tokens_accumulated = 0


    # Initialize a list to store the full messages
    full_message_list = []
    # If this is not a continue request, we add the AI prompt
    message_tokenized = lollms_app.model.tokenize(
        "\n" +lollms_app.personality.ai_message_prefix.strip()
    )
    full_message_list.append(message_tokenized)
    # Update the cumulative number of tokens
    tokens_accumulated += len(message_tokenized)


    message_tokenized = lollms_app.model.tokenize(discussion)
    if len(message_tokenized)>lollms_app.config.ctx_size-1024:
        pos = message_tokenized[-(lollms_app.config.ctx_size-1024)]
        detokenized = lollms_app.model.detokenize(message_tokenized[pos:pos+20])
        position = discussion.find(detokenized)
        if position!=-1 and position>0:
            discussion_messages = discussion[-position:]
        else:
            discussion_messages = discussion[-(lollms_app.config.ctx_size-1024):]
    else:
        discussion_messages = discussion

    # Build the final prompt by concatenating the conditionning and discussion messages
    prompt_data = conditionning + documentation + history + user_description + discussion_messages

    # Tokenize the prompt data
    tokens = lollms_app.model.tokenize(prompt_data)

    # if this is a debug then show prompt construction details
    if lollms_app.config["debug"]:
        ASCIIColors.bold("CONDITIONNING")
        ASCIIColors.yellow(conditionning)
        ASCIIColors.bold("DOC")
        ASCIIColors.yellow(documentation)
        ASCIIColors.bold("HISTORY")
        ASCIIColors.yellow(history)
        ASCIIColors.bold("DISCUSSION")
        ASCIIColors.hilight(discussion_messages,"!@>",ASCIIColors.color_yellow,ASCIIColors.color_bright_red,False)
        ASCIIColors.bold("Final prompt")
        ASCIIColors.hilight(prompt_data,"!@>",ASCIIColors.color_yellow,ASCIIColors.color_bright_red,False)
        ASCIIColors.info(f"prompt size:{len(tokens)} tokens") 
        ASCIIColors.info(f"available space after doc and history:{available_space} tokens") 

    # Return the prepared query, original message content, and tokenized query
    return prompt_data

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.content.startswith(config["summoning_word"]):
        prompt = message.content[len(config["summoning_word"])+1:]

        context['discussion'] = prepare_query(context['discussion'], prompt, 1024)
        context['discussion']+= "\n!@>" + message.author.name +": "+  prompt + "\n" + f"{lollms_app.personality.ai_message_prefix}"
        context['current_response']=""
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
        lollms_app.safe_generate(context['discussion'], n_predict=1024, callback=callback, placeholder={"discussion":context['discussion']},place_holders_to_sacrifice=["discussion"], debug=True)

        context['discussion'] += context['current_response']
        await message.channel.send(context['current_response'])
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
