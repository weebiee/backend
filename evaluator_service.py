import Evaluator_pb2_grpc as pb
from model import Evaluator, Evaluation, Sentiment
from Evaluator_pb2_grpc import EvaluatorServicer


def _get_score_from_evaluation(evaluation: Evaluation):
    return pb.Score(
        positivity=evaluation[Sentiment.POSITIVE],
        negativity=evaluation[Sentiment.NEGATIVE],
        neutrality=evaluation[Sentiment.NEUTRAL]
    )


class EvaluatorServicerImpl(EvaluatorServicer):
    def __init__(self, evaluator: Evaluator):
        self.__evaluator = evaluator

    async def GetScores(self, request, context):
        evals = await self.__evaluator.evaluate(list(p.content for p in request.phrases))
        return pb.GetScoresResponse(
            ok=True,
            scores=list(_get_score_from_evaluation(e) for e in evals)
        )
