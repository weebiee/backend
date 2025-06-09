import uuid

import torch.cuda

import Evaluator_pb2 as _pb
from Evaluator_pb2_grpc import EvaluatorServicer
from model import Evaluator, Evaluation, Sentiment, MLEvaluator


def _get_score_from_evaluation(evaluation: Evaluation):
    return _pb.Score(
        positivity=evaluation[Sentiment.POSITIVE],
        negativity=evaluation[Sentiment.NEGATIVE],
        neutrality=evaluation[Sentiment.NEUTRAL]
    )


class EvaluatorServicerImpl(EvaluatorServicer):
    def __init__(self, evaluator: Evaluator):
        self.__evaluator = evaluator
        self.__tasks_count = 0
        self.__id = str(uuid.uuid4())

    async def Heartbeat(self, request, context):
        if isinstance(self.__evaluator, MLEvaluator) and self.__evaluator.model.device.type == 'cuda':
            device = self.__evaluator.model.device
            total = torch.cuda.get_device_properties(device).total_memory >> 10
            free = torch.cuda.memory_usage(device) >> 10
        else:
            import psutil
            vram = psutil.virtual_memory()
            total, free = vram.total, vram.free
        return _pb.HeartbeatResponse(
            tasks=self.__tasks_count,
            free_vram=free,
            total_vram=total,
            id=self.__id
        )

    async def GetScores(self, request, context):
        self.__tasks_count += len(request.phrases)
        try:
            evals = await self.__evaluator.evaluate(list(p.content for p in request.phrases))
            return _pb.GetScoresResponse(
                ok=True,
                scores=list(_get_score_from_evaluation(e) for e in evals)
            )
        except Exception as e:
            return _pb.GetScoresResponse(
                ok=False,
                err_msg=str(e),
                scores=list()
            )
        finally:
            self.__tasks_count -= len(request.phrases)
