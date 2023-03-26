from app.model_instance import ModelInstance

class MockModel:
    def __init__(self, mock_return):
        self.mock_return = mock_return

    def predict(self, model_inputs):
        return self.mock_return

def mock_classes_parser(classNum):
    return "easy" if classNum == 0 else "hard"

def test_model_instance():
    model = MockModel([])
    feature_names = ['meds', 'comps']
    classes_parser = mock_classes_parser

    model_instance = ModelInstance(model, feature_names, classes_parser)

    assert model_instance.feature_names == feature_names
    assert model_instance.model == model
    assert model_instance.classes_parser == classes_parser

def test_model_instance_parse_samples_success():
    model = None
    feature_names = ['meds', 'comps']
    classes_parser = None

    model_instance = ModelInstance(model, feature_names, classes_parser)

    samples = [
        {
            "meds": 1,
            "comps": 2
        },
        {
            "meds": 2,
            "comps": 3
        }
    ]

    model_inputs, errors = model_instance.parse_samples(samples)

    assert model_inputs == [[1, 2], [2, 3]]
    assert errors == []

def test_model_instance_parse_samples_error():
    model = None
    feature_names = ['meds', 'comps', 'doses']
    classes_parser = None

    model_instance = ModelInstance(model, feature_names, classes_parser)

    samples = [
        {
            "meds": 5,
            "comps": 7
        },
        {
            "meds": 4,
            "comps": 2.5,
            "doses": 1
        },
        {
            "meds": 6
        }
    ]

    model_inputs, errors = model_instance.parse_samples(samples)

    assert model_inputs == [[4, 2.5, 1]]

    # (idx, error_message)
    assert errors == [(0, "Missing features: doses"), (2, "Missing features: comps, doses")]

def test_model_instance_predic():
    model = MockModel([2,1])
    feature_names = ['meds', 'comps']
    classes_parser = None

    model_instance = ModelInstance(model, feature_names, classes_parser)

    model_inputs = [[2, 3], [1, 2]]

    predicts = model_instance.predict(model_inputs)

    assert predicts == [2, 1]

def test_model_instance_predic_empty():
    model = MockModel(['will no be called'])
    feature_names = ['meds', 'comps']
    classes_parser = None

    model_instance = ModelInstance(model, feature_names, classes_parser)

    model_inputs = []

    predicts = model_instance.predict(model_inputs)

    assert predicts == []

def test_model_instance_predict_parser():
    model = MockModel([1, 0, 1])
    feature_names = ['meds', 'comps']
    classes_parser = mock_classes_parser

    model_instance = ModelInstance(model, feature_names, classes_parser)

    model_inputs = [[5, 7], [2, 3], [4, 5]]

    predicts = model_instance.predict(model_inputs)

    assert predicts == ['hard', 'easy', 'hard']

def mock_sample_transform(sample):
    _sample = sample.copy()
    _sample['meds'] /= 2
    return _sample

def test_model_instance_sample_transform():
    model = None
    feature_names = ['meds', 'comps']
    classes_parser = None
    sample_transform = mock_sample_transform

    model_instance = ModelInstance(model, feature_names, classes_parser, sample_transform)

    samples = [
        {
            "comps": 7,
            "meds": 4
        },
        {
            "comps": 9,
            "meds": 6
        }
    ]

    model_inputs, errors = model_instance.parse_samples(samples)

    assert model_inputs == [[2, 7], [3, 9]]
    assert errors == []