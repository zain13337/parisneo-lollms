from lollms.console import Conversation 

class MyConversation(Conversation):
  def __init__(self, cfg=None):
    super().__init__(cfg, show_welcome_message=False)

  def start_conversation(self):
    prompt = "Once apon a time"
    def callback(text, type=None):
        print(text, end="", flush=True)
        return True
    print(prompt, end="", flush=True)
    output = self.safe_generate(prompt, callback=callback)

if __name__ == '__main__':
  cv = MyConversation()
  cv.start_conversation()