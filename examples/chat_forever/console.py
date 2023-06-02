from lollms.console import Conversation 

class MyConversation(Conversation):
  def __init__(self, cfg=None):
    super().__init__(cfg, show_welcome_message=False)

  def start_conversation(self):
    full_discussion=""
    while True:
      prompt = input("You: ")
      if prompt=="exit":
        return
      if prompt=="menu":
        self.menu.main_menu()
      full_discussion += self.personality.user_message_prefix+prompt+self.personality.link_text
      full_discussion += self.personality.ai_message_prefix
      def callback(text, type=None):
          print(text, end="", flush=True)
          return True
      print(self.personality.name+": ",end="",flush=True)
      output = self.safe_generate(full_discussion, callback=callback)
      full_discussion += output.strip()+self.personality.link_text
      print()

if __name__ == '__main__':
  cv = MyConversation()
  cv.start_conversation()