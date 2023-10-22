import discord
from discord.ext import commands
from lollms.apps.console import Conversation
import yaml
from pathlib import Path

config_path = Path(__file__).resolve().parent / 'discord_config.yaml'

if not config_path.exists():
    bot_token = input("Please enter your bot token: ")
    config = {'bot_token': bot_token}
    with open(config_path, 'w') as config_file:
        yaml.safe_dump(config, config_file)
else:
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    bot_token = config['bot_token']

bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())

class ChatBot(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.cv = Conversation()
        self.context = {}

    @commands.command()
    async def chat(self, ctx, *, prompt):
        self.context[ctx.author.id] = f'{self.context.get(ctx.author.id, "")}:{prompt}\nlollms_chatbot:'
        def callback(text, type=None):
            self.context[ctx.author.id] += text
            return True
        self.cv.safe_generate(self.context[ctx.author.id], callback=callback)
        await ctx.send(self.context[ctx.author.id])
    @commands.command()
    async def install(self, ctx):
        response = "To install lollms, make sure you have installed the currently supported python version (consult the repository to verify what version is currently supported, but as of 10/22/2023, the version is 3.10).\nThen you can follow these steps:\n1. Open your command line interface.\n2. Navigate to the directory where you want to install lollms.\n3. Run the following command to clone the lollms repository: `git clone https://github.com/lollms/lollms.git`.\n4. Once the repository is cloned, navigate into the lollms directory.\n5. Run `pip install -r requirements.txt` to install the required dependencies.\n6. You can now use lollms by importing it in your Python code."
    @commands.Cog.listener()
    async def on_member_join(self, member):
        channel = member.guild.system_channel
        if channel is not None:
            await channel.send(f'Welcome {member.mention} to the server! How can I assist you today?')
@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')
    print('------')

bot.add_cog(ChatBot(bot))
bot.run(bot_token)
