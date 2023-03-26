import os

from os_io import load_model
from model_instance import ModelInstance

default_model_name = '../data/classifierComplexidadeCPM.pkl'

feature_names = ['comps', 'qtdHrs', 'meds', 'doses', 'valor']

def transform_sample(sample):
    transformed = sample.copy()
    if 'comps' in sample and 'percFora' in sample:
        transformed['comps'] *= (1-(transformed['percFora']/100))
    return transformed

def parser_classes(class_label):
    if class_label == 0:
        return "fácil"
    return "médio" if class_label == 2 else "difícil"

def get_cpm_classifier():
    model_name = os.environ['MODEL_NAME'] if 'MODEL_NAME' in os.environ  else default_model_name
    model = load_model(model_name)

    model_instance = ModelInstance(model, feature_names, parser_classes, transform_sample)

    return model_instance