class ModelInstance:
    def __init__(self, model, feature_names, classes_parser=None, sample_transformer=None):
        self.feature_names = feature_names
        self.model = model
        self.classes_parser = classes_parser
        self.transformer = sample_transformer or (lambda x: x)
    
    def parse_samples(self, samples: list[dict]):
        good_samples, errors = self.__check_feature_names(samples)
        
        transformed_samples = list(map(lambda s: self.transformer(s), good_samples))

        model_inputs = list(map(lambda s: [s[name] for name in self.feature_names], transformed_samples)) 

        return model_inputs, errors

    def predict(self, model_inputs):
        if len(model_inputs) < 1:
            return []
        classes = self.model.predict(model_inputs)
        if (self.classes_parser):
            return list(map(self.classes_parser, classes))
        return classes

    def __check_feature_names(self, samples):
        features_set = set(self.feature_names)

        errors: list[tuple[int, str]] = []
        good_samples = []

        for idx, sample in enumerate(samples):
            missing_features = features_set - set(sample)
            if missing_features:
                errors.append((idx, f"Missing features: {', '.join(sorted(missing_features))}"))
            else:
                good_samples.append(sample)
        
        return good_samples, errors