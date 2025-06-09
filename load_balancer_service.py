import asyncio
import time
import uuid
from asyncio import TaskGroup
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Iterable

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
    idle_vram: int = 0
    last_evaluation: _pb.LastExecution | None = None
    id: str | None = None


@asynccontextmanager
async def load_balancer_servicer(subnodes: list[str], channel_credentials: grpc.ChannelCredentials | None = None):
    impl = LoadBalancerServicerImpl(subnodes, channel_credentials)
    yield impl
    await impl.close()


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
            grpc.secure_channel(addr, channel_credentials) if channel_credentials else grpc.aio.insecure_channel(addr)
            for addr in subnodes)
        self.__last_refresh = 0
        self.__id = str(uuid.uuid4().hex)

    @property
    def id(self) -> str:
        return self.__id

    async def refresh(self, exception_list: Iterable[str] | None = None, force: bool = False) -> list[Subnode]:
        if not force and time.time() - self.__last_refresh < 30:
            return self.__subnodes

        for (node, channel) in zip(self.__subnodes, self.__channels):
            if exception_list and node.address in exception_list:
                continue

            stub = _rpc.EvaluatorStub(channel)
            try:
                res = await stub.Heartbeat(_pb.HeartbeatRequest())
            except grpc.RpcError as e:
                raise SubnodeUnavailableError(address=node.address, inner=e)

            node.tasks = res.tasks
            node.free_vram = res.free_vram
            node.total_vram = res.total_vram
            node.id = res.id
            node.last_evaluation = res.last_evaluation

            if res.id == self.__id:
                raise SubnodeUnavailableError(address=node.address, inner='subnode list contains this load balancer')

            if res.tasks == 0:
                node.idle_vram = res.total_vram - res.free_vram

        self.__last_refresh = time.time()
        return self.__subnodes

    async def close(self):
        for chan in self.__channels:
            await chan.close()

    def __get_best_node_stub(self, new_tasks_count: int,
                             exception_list: Iterable[str] | None = None) \
            -> list[tuple[_rpc.EvaluatorStub, int]] | None:
        import heapq

        prediction_list = []
        for idx, node in enumerate(self.__subnodes):
            if node.id is None or exception_list and node.address in exception_list:
                continue
            mem_per_task = (node.total_vram - node.free_vram - node.idle_vram) / node.tasks if node.tasks > 0 \
                else (node.total_vram - node.last_evaluation.free_vram - node.idle_vram) / node.last_evaluation.tasks \
                if node.last_evaluation and node.last_evaluation.tasks > 0 else max(node.free_vram, 1)
            predicted_free_mem = node.free_vram - mem_per_task * new_tasks_count
            heapq.heappush(prediction_list, (predicted_free_mem, mem_per_task, idx))

        if not prediction_list:
            return None

        allocation_list = []
        allocated_task = 0
        for (_, mem_per_task, idx) in prediction_list:
            node = self.__subnodes[idx]
            due_tasks = int(node.free_vram // mem_per_task) if mem_per_task > 0 else new_tasks_count - allocated_task
            allocated_task += due_tasks
            allocation_list.append((_rpc.EvaluatorStub(self.__channels[idx]), due_tasks))
            if allocated_task > new_tasks_count:
                break

        return allocation_list

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
        results = []
        phrases: list[str] = request.phrases

        while len(results) < len(phrases):
            exception_set = set()
            try:
                await self.refresh(exception_set, force=bool(results))
            except SubnodeUnavailableError as e:
                exception_set.add(e.address)

            allocation = self.__get_best_node_stub(new_tasks_count=len(request.phrases),
                                                   exception_list=exception_set)
            tasks: list[asyncio.Task] = []
            while phrases and allocation:
                (stub, chunk_size) = allocation.pop()
                chunk = phrases[:chunk_size]
                phrases = phrases[chunk_size:]

                async def send_rpc_get_score():
                    return await stub.GetScores(_pb.GetScoresRequest(phrases=chunk))

                tasks.append(asyncio.Task(send_rpc_get_score()))

            for task in tasks:
                await task
                res = task.result()
                if not res.ok:
                    return res

                scores = res.scores
                results += scores

        return res
