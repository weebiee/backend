import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass

import grpc

from Evaluator_pb2_grpc import EvaluatorServicer
import Evaluator_pb2_grpc as _rpc
import Evaluator_pb2 as _pb


@dataclass
class Subnode:
    address: str
    tasks: int = -1
    free_vram: int = -1
    total_vram: int = -1
    memory_offset: int = 0
    id: str | None = None


@asynccontextmanager
async def load_balancer_servicer(subnodes: list[str], channel_credentials: grpc.ChannelCredentials | None = None):
    impl = LoadBalancerServicerImpl(subnodes, channel_credentials)
    yield impl
    impl.close()


class SubnodeUnavailableError(Exception):
    def __init__(self, address: str, inner: Exception | str, *args):
        self.address = address
        self.inner = inner
        super().__init__(*args)

    def __str__(self):
        return f'Subnode {self.address} is unavailable: {self.inner}'


class LoadBalancerServicerImpl(EvaluatorServicer):
    def __init__(self, subnodes: list[str], channel_credentials: grpc.ChannelCredentials | None = None):
        self.__subnodes = list(Subnode(address=addr) for addr in subnodes)
        self.__channels = list(
            grpc.secure_channel(addr, channel_credentials) if channel_credentials else grpc.aio.insecure_channel(addr) for
            addr in subnodes)
        self.__last_refresh = 0
        self.__id = str(uuid.uuid4().hex)

    @property
    def id(self) -> str:
        return self.__id

    async def refresh(self) -> list[Subnode]:
        if time.time() - self.__last_refresh < 30:
            return self.__subnodes

        for (node, channel) in zip(self.__subnodes, self.__channels):
            stub = _rpc.EvaluatorStub(channel)
            try:
                res = await stub.Heartbeat(_pb.HeartbeatRequest())
            except grpc.RpcError as e:
                raise SubnodeUnavailableError(address=node.address, inner=e)

            node.tasks = res.tasks
            node.free_vram = res.free_vram
            node.total_vram = res.total_vram
            node.id = res.id

            if res.id == self.__id:
                raise SubnodeUnavailableError(address=node.address, inner='subnode list contains this load balancer')

            if res.tasks <= 3:
                node.memory_offset = res.free_vram - res.total_vram

        self.__last_refresh = time.time()
        return self.__subnodes

    def close(self):
        for chan in self.__channels:
            chan.close()

    def __get_best_node_stub(self, new_tasks_count: int) -> _rpc.EvaluatorStub | None:
        from math import inf
        best_chan: grpc.Channel | None = None
        best_prediction = inf

        for idx, node in enumerate(self.__subnodes):
            if node.id is None:
                continue

            mem_per_task = (node.total_vram - node.free_vram + node.memory_offset) / node.tasks
            predicted_free_mem = node.free_vram - mem_per_task * new_tasks_count
            if best_prediction > predicted_free_mem > 0:
                best_chan = self.__channels[idx]
                best_prediction = predicted_free_mem

        if not best_chan:
            return None

        return _rpc.EvaluatorStub(best_chan)

    async def Heartbeat(self, request, context):
        await self.refresh()

        tasks = 0
        free_vram = 0
        total_vram = 0
        for node in self.__subnodes:
            tasks += node.tasks
            free_vram += node.free_vram
            total_vram += node.total_vram

        return _pb.HeartbeatResponse(
            tasks=tasks,
            free_vram=free_vram,
            total_vram=total_vram,
            id=self.__id
        )

    async def GetScores(self, request, context):
        stub = self.__get_best_node_stub(new_tasks_count=len(request.phrases))
        if stub:
            return await stub.GetScores(_pb.GetScoresRequest(phrases=request.phrases))
        else:
            return _pb.GetScoresResponse(
                ok=False,
                err_msg='No available subnode.',
                scores=list()
            )
