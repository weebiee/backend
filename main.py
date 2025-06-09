import asyncio
import re
from argparse import ArgumentParser

import grpc.aio

import Evaluator_pb2_grpc as _rpc


def check_token_validity(token: str) -> bool:
    return re.match(r'^[a-zA-Z_]{13,}$', token) is not None


def make_server(args, server: grpc.aio.Server):
    if args.private_key and args.certificate_chain:
        server.add_secure_port(
            address=args.address,
            server_credentials=grpc.ssl_server_credentials([(args.private_key, args.certificate_chain)])
        )
    else:
        server.add_insecure_port(address=args.address)


async def _await_cancellation():
    try:
        while True:
            await asyncio.sleep(100)
    except asyncio.CancelledError:
        return


async def serve_load_balancer(args):
    if not args.subnodes:
        from sys import stderr
        print('Warning: no subnodes registered. This load balancer is rendered useless.', file=stderr)

    from load_balancer_service import load_balancer_servicer, SubnodeUnavailableError
    async with load_balancer_servicer(args.subnodes,
                                      grpc.ssl_channel_credentials() if args.secure_subnodes else None) as servicer:

        try:
            online = len(await servicer.refresh())
            print(f'{online} node(s) online.')
        except SubnodeUnavailableError as e:
            from sys import stderr
            print(f'Warning: subnode {e.address} is unavailable: {e.inner}', file=stderr)

        server = grpc.aio.server()
        make_server(args, server)
        _rpc.add_EvaluatorServicer_to_server(servicer, server)
        await server.start()
        await _await_cancellation()
        await server.stop(grace=10)


async def serve_evaluator(args):
    from evaluator_service import EvaluatorServicerImpl
    from model import MLEvaluator

    evaluator = MLEvaluator(base_model_name='Alibaba-NLP/gte-Qwen2-1.5B-instruct', addition_path='ml/addition.pt')
    servicer = EvaluatorServicerImpl(evaluator)

    server = grpc.aio.server()
    make_server(args, server)

    _rpc.add_EvaluatorServicer_to_server(servicer, server)
    await server.start()

    await _await_cancellation()
    await server.stop(grace=10)


async def main():
    parser = ArgumentParser()
    parser.add_argument('--load-balancer', '-L', action='store_true', default=False, help='Work as a load balancer, '
                                                                                          'instead of evaluator.')
    parser.add_argument('--address', '-A', nargs='?', type=str, default='[::]:63398')
    parser.add_argument('--token', type=str, required=True)
    parser.add_argument('--private-key', '--pk', type=str)
    parser.add_argument('--certificate-chain', '--ch', type=str)
    parser.add_argument('--secure-subnodes', '-S', action='store_true', default=False)
    parser.add_argument('subnodes', nargs='*', default=list())

    args = parser.parse_args()

    if not check_token_validity(args.token):
        from sys import stderr
        print('Invalid token. Only Latin letters and underscores are permitted.', file=stderr)
        return

    if not (args.private_key and args.certificate_chain) and (args.private_key or args.certificate_chain):
        from sys import stderr
        print('Missing either of private key or certificate chain.', file=stderr)
        return

    if args.load_balancer:
        await serve_load_balancer(args)
    else:
        await serve_evaluator(args)


if __name__ == '__main__':
    asyncio.run(main())
