from app.response import parse_model_response

def test_parse_model_response_ordered():
    predicts = ["easy", "hard", "hard"]
    errors = [(0, "error happened"), (2, "another error")]

    response = parse_model_response(predicts, errors)

    assert response == [
        {"error": "error happened"},
        {"predict": "easy"},
        {"error": "another error"},
        {"predict": "hard"},
        {"predict": "hard"}
    ]

def test_parse_model_response_unordered():
    predicts = ["easy", "hard", "hard"]
    errors = [(2, "another error"), (0, "error happened")]

    response = parse_model_response(predicts, errors)

    assert response == [
        {"error": "error happened"},
        {"predict": "easy"},
        {"error": "another error"},
        {"predict": "hard"},
        {"predict": "hard"}
    ]

def test_parse_model_response_bounderies():
    predicts = ["easy"]
    errors = [(0, "'doses' expected"), (2, "'comps, doses' expected")]

    response = parse_model_response(predicts, errors)

    assert response == [
        {"error": "'doses' expected"},
        {"predict": "easy"},
        {"error": "'comps, doses' expected"}
    ]

    