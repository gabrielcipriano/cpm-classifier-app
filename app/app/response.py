def parse_model_response(predicts: list, errors: list[tuple[int, str]]):

    response = list(map(lambda p: {"predict": p}, predicts))

    for idx, error_message in sorted(errors, key=lambda e: e[0]):
        response.insert(idx, {"error": error_message})
    
    return response