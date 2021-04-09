from sklearn.preprocessing import StandardScaler, QuantileTransformer

class Learning:

    def __init__(self) -> None:
        self.pipe = None

    def __setattr__(self, name: str, value: any) -> None:
        self.__dict__[name] = value
    
    def scale(self):
        pass


if __name__ == "__main__":
    pass
 