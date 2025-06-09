from enum import Enum
from os import PathLike


class Sentiment(Enum):
    UNKNOWN = -1
    POSITIVE = 0
    NEGATIVE = 1
    NEUTRAL = 2


class Evaluation:
    def __init__(self, source: dict[Sentiment | int | str, float]):
        self.__confidences = dict(
            (Sentiment(key) if isinstance(key, int) else Sentiment[key] if isinstance(key, str) else key, float(value))
            for (key, value) in source.items())

    def __getitem__(self, item: Sentiment):
        return self.__confidences[item]

    def __setitem__(self, key: Sentiment, value: float):
        return self.__confidences


class Phrase:
    def __init__(self, content: str, evaluation: Evaluation | None = None):
        self.content = content
        self.evaluation = evaluation


class Evaluator:
    async def evaluate(self, phrases: list[Phrase]) -> list[Evaluation]:
        pass


class MLEvaluator(Evaluator):
    def __init__(self, base_model_name: str, addition_path: str | PathLike[str]):
        self.base_model_name = base_model_name
        self.addition_path = addition_path

        from ml.model import EmbeddingModel
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        self.model = EmbeddingModel(self.base_model_name, 3, addition_path=self.addition_path).eval()

    async def evaluate(self, phrases: list[Phrase | str]) -> list[Evaluation]:
        if len(phrases) <= 0:
            return []
        if type(phrases[0]) is Phrase:
            phrase_tokenizing = list(v.content for v in phrases)
        else:
            phrase_tokenizing = phrases

        tokens = self.tokenizer(phrase_tokenizing, padding=True, truncation=True, return_tensors='pt').to(
            self.model.device)
        scores = self.model(**tokens).cpu().detach().numpy()

        evals = list(Evaluation(dict((Sentiment(idx), s) for (idx, s) in enumerate(score))) for score in scores)
        if type(phrases[0]) is Phrase:
            for idx, e in enumerate(evals):
                phrases[idx].evaluation = e

        return evals
