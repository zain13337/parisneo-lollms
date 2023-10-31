import discord
from discord.ext import commands
from lollms.app import LollmsApplication
from lollms.paths import LollmsPaths
from lollms.main_config import LOLLMSConfig
import yaml
from pathlib import Path
from ascii_colors import ASCIIColors
from safe_store import TextVectorizer, GenericDataLoader, VisualizationMethod, VectorizationMethod

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
            docs = "!@>Documentation:\n"+'\n'.join(docs)
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
