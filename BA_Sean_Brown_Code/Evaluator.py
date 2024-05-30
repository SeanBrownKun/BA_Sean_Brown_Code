import Levenshtein


class Evaluator:
    """This class calculates the Levensthein distance, either
    on word level or on character level."""

    def __init__(self):
        self.lsd = Levenshtein.distance

    def evaluate_one_example(self, outputs, org, word_level=False):
        if word_level:
            if type(outputs) == list:
                result = [(out, self.lsd(out.split(), org.split())) for out in outputs]
            else:
                result = (outputs, self.lsd(outputs.split(), org.split()))
        else:
            if type(outputs) == list:
                result = [(out, self.lsd(out, org)) for out in outputs]
            else:
                result = (outputs, self.lsd(outputs, org))
        # result.sort(key=lambda x: x[1])
        return result



