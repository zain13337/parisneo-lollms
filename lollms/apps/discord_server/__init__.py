import discord
from discord.ext import commands
from lollms.apps.console import Conversation

bot = commands.Bot(command_prefix='!')

class ChatBot(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.cv = Conversation()

    @commands.command()
    async def chat(self, ctx, *, prompt):
        full_discussion = ""
        def callback(text, type=None):
            full_discussion += text
            return True
        self.cv.safe_generate(full_discussion, callback=callback)
        await ctx.send(full_discussion)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')
    print('------')

bot.add_cog(ChatBot(bot))
bot.run('YOUR_BOT_TOKEN')
