from enum import Enum
from os import PathLike


class Sentiment(Enum):
    UNKNOWN = -1
    POSITIVE = 0
    NEGATIVE = 1
    NEUTRAL = 2


class Evaluation:
    def __init__(self, source: dict[Sentiment | int | str, float]):
        if any(not isinstance(value, float) and not isinstance(value, int) for (_, value) in source):
            raise ValueError('source')

        self.__confidences = dict(
            (Sentiment(key) if isinstance(key, int) else Sentiment[key] if isinstance(key, str) else key, value) for
            (key, value) in source)

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
    def __init__(self, base_model_name: str, addition_file_path: str | PathLike[str]):
        self.base_model_name = base_model_name
        self.addition_file_path = addition_file_path

    async def evaluate(self, phrases: list[Phrase]) -> list[Evaluation]:
        from ml.model import EmbeddingModel
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        model = EmbeddingModel(self.base_model_name, 3, addition_path=self.addition_file_path)


