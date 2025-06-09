import torch.cuda

import Evaluator_pb2_grpc as pb
from Evaluator_pb2_grpc import EvaluatorServicer
from model import Evaluator, Evaluation, Sentiment, MLEvaluator


def _get_score_from_evaluation(evaluation: Evaluation):
    return pb.Score(
        positivity=evaluation[Sentiment.POSITIVE],
        negativity=evaluation[Sentiment.NEGATIVE],
        neutrality=evaluation[Sentiment.NEUTRAL]
    )


class EvaluatorServicerImpl(EvaluatorServicer):
    def __init__(self, evaluator: Evaluator):
        self.__evaluator = evaluator
        self.__tasks_count = 0

    async def Heartbeat(self, request, context):
        if isinstance(self.__evaluator, MLEvaluator) and self.__evaluator.model.device.startswith('cuda'):
            device = self.__evaluator.model.device
            total = torch.cuda.get_device_properties(device).total_memory
            free = torch.cuda.memory_usage(device)
        else:
            import psutil
            vram = psutil.virtual_memory()
            total, free = vram.total, vram.free
        return pb.HeartbeatResponse(
            tasks=self.__tasks_count,
            free_vram=free,
            total_vram=total
        )

    async def GetScores(self, request, context):
        self.__tasks_count += 1
        try:
            evals = await self.__evaluator.evaluate(list(p.content for p in request.phrases))
            return pb.GetScoresResponse(
                ok=True,
                scores=list(_get_score_from_evaluation(e) for e in evals)
            )
        except Exception as e:
            return pb.GetScoresResponse(
                ok=False,
                err_msg=str(e),
                scores=list()
            )
        finally:
            self.__tasks_count -= 1
