import pickle
import os

def load_model(filename: str):
    path = os.path.dirname(__file__)
    filepath = os.path.join(path, filename)
    model = pickle.load(open(filepath, 'rb'))
    return model