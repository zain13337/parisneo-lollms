

class ASCIIColors:
    # Reset
    color_reset = '\u001b[0m'

    # Regular colors
    color_black = '\u001b[30m'
    color_red = '\u001b[31m'
    color_green = '\u001b[32m'
    color_yellow = '\u001b[33m'
    color_blue = '\u001b[34m'
    color_magenta = '\u001b[35m'
    color_cyan = '\u001b[36m'
    color_white = '\u001b[37m'
    color_orange = '\u001b[38;5;202m'

    # Bright colors
    color_bright_black = '\u001b[30;1m'
    color_bright_red = '\u001b[31;1m'
    color_bright_green = '\u001b[32;1m'
    color_bright_yellow = '\u001b[33;1m'
    color_bright_blue = '\u001b[34;1m'
    color_bright_magenta = '\u001b[35;1m'
    color_bright_cyan = '\u001b[36;1m'
    color_bright_white = '\u001b[37;1m'
    color_bright_orange = '\u001b[38;5;208m'

    @staticmethod
    def print(text, color=color_bright_red, end="\n", flush=False):
        print(f"{color}{text}{ASCIIColors.color_reset}", end=end, flush=flush)

    @staticmethod
    def warning(text, end="\n", flush=False):
        print(f"{ASCIIColors.color_bright_orange}{text}{ASCIIColors.color_reset}", end=end, flush=flush)

    @staticmethod
    def error(text, end="\n", flush=False):
        print(f"{ASCIIColors.color_bright_red}{text}{ASCIIColors.color_reset}", end=end, flush=flush)

    @staticmethod
    def success(text, end="\n", flush=False):
        print(f"{ASCIIColors.color_green}{text}{ASCIIColors.color_reset}", end=end, flush=flush)

    @staticmethod
    def info(text, end="\n", flush=False):
        print(f"{ASCIIColors.color_blue}{text}{ASCIIColors.color_reset}", end=end, flush=flush)

    @staticmethod
    def red(text, end="\n", flush=False):
        print(f"{ASCIIColors.color_red}{text}{ASCIIColors.color_reset}", end=end, flush=flush)
    @staticmethod
    def green(text, end="\n", flush=False):
        print(f"{ASCIIColors.color_green}{text}{ASCIIColors.color_reset}", end=end, flush=flush)

    @staticmethod
    def blue(text, end="\n", flush=False):
        print(f"{ASCIIColors.color_blue}{text}{ASCIIColors.color_reset}", end=end, flush=flush)

    @staticmethod
    def yellow(text, end="\n", flush=False):
        print(f"{ASCIIColors.color_yellow}{text}{ASCIIColors.color_reset}", end=end, flush=flush)

    @staticmethod
    def magenta(text, end="\n", flush=False):
        print(f"{ASCIIColors.color_magenta}{text}{ASCIIColors.color_reset}", end=end, flush=flush)

    @staticmethod
    def cyan(text, end="\n", flush=False):
        print(f"{ASCIIColors.color_cyan}{text}{ASCIIColors.color_reset}", end=end, flush=flush)

    

