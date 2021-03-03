class Indenter:
    def __init__(self):
        self.padding = 0

    def __enter__(self):
        self.padding += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.padding -= 1

    def print(self, text):
        print('\t' * self.padding + text)

with Indenter() as indent:
    indent.print("hi!")
    with indent:
        indent.print("hello")
        with indent:
            indent.print("bonjour")
    indent.print("hey")
