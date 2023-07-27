from lollms.apps.console import Conversation 
import sys

class MyConversation(Conversation):
  def __init__(self, cfg=None):
    super().__init__(cfg, show_welcome_message=False)

  def start_conversation(self):
    prompt = "Once apon a time"
    def callback(text, type=None, metadata:dict={}):
        print(text, end="")
        sys.stdout = sys.__stdout__
        sys.stdout.flush()

        return True
    print(prompt, end="")
    sys.stdout = sys.__stdout__
    sys.stdout.flush()
    output = self.safe_generate(prompt, callback=callback)

if __name__ == '__main__':
  cv = MyConversation()
  cv.start_conversation()