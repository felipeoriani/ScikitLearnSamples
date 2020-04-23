import platform

class Util:
    @staticmethod
    def clear(self):
        system = platform.system().lower()
        if system == 'windows':
            platform.os.system('cls')
        else:
            platform.os.system('clear')