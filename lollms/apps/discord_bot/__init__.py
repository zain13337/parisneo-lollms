import discord
from discord.ext import commands
from lollms.app import LollmsApplication
from lollms.paths import LollmsPaths
from lollms.main_config import LOLLMSConfig
import yaml
from pathlib import Path
from ascii_colors import ASCIIColors
from safe_store import TextVectorizer, GenericDataLoader, VisualizationMethod, VectorizationMethod

contexts={}

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
        prompt = message.content[len(config["summoning_word"]):]
        print("Chatting")
        try:
            docs, _ = text_vectorzer.recover_text(prompt,3)
            docs = "!#>Documentation:\n"+'\n'.join(docs)
        except:
            docs=""
        context = f"""!#>instruction:
{lollms_app.personality.personality_conditioning}
!#>Informations:
Current model:{lollms_app.config.model_name}
Current personality:{lollms_app.personality.name}
{docs}
!#>{message.author.id}: {prompt}
!#>{lollms_app.personality.ai_message_prefix}: """
        print("Context:"+context)
        out = ['']
        def callback(text, type=None):
            out[0] += text
            print(text,end="")
            return True
        
        ASCIIColors.green("Warming up")
        lollms_app.safe_generate(context, n_predict=1024, callback=callback)
        print()
        await message.channel.send(out[0])
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
