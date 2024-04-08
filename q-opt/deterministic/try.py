class A:
    def __init__(self) -> None:
        self.name = "a"
    
    def print_name(self):
        print("a")

class B(A):
    def __init__(self) -> None:
        super(B, self).__init__()
        self.print_name()

b = B()