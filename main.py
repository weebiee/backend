import asyncio
import re
from argparse import ArgumentParser

import grpc.aio


def check_token_validity(token: str) -> bool:
    return re.match(r'^[a-zA-Z_]{13,}$', token) is not None


async def serve_load_balancer(args):
    pass


async def serve_evaluator(args):
    import Evaluator_pb2_grpc as rpc
    from evaluator_service import EvaluatorServicerImpl
    from model import MLEvaluator

    evaluator = MLEvaluator(base_model_name='Alibaba-NLP/gte-Qwen2-1.5B-instruct', addition_path='ml/addition.pt')
    servicer = EvaluatorServicerImpl(evaluator)

    server = grpc.aio.server()
    rpc.add_EvaluatorServicer_to_server(servicer, server)
    if args.private_key and args.certificate_chain:
        server.add_secure_port(
            address=args.address,
            server_credentials=grpc.ssl_server_credentials([(args.private_key, args.certificate_chain)])
        )
    else:
        server.add_insecure_port(address=args.address)

    await server.start()
    await server.wait_for_termination()


async def main():
    parser = ArgumentParser()
    parser.add_argument('--load-balancer', '-L', nargs='?', default=False, help='Work as a load balancer, '
                                                                                'instead of evaluator.')
    parser.add_argument('--address', '-A', nargs='?', type=str, default='[::]:69283')
    parser.add_argument('--token', type=str, required=True)
    parser.add_argument('--private-key', '--pk', type=str)
    parser.add_argument('--certificate-chain', '--ch', type=str)

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
