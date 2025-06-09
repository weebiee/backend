# Evaluator and Load Balancer

## Getting started

This project is managed by `uv`. Run
```shell
uv sync && uv venv
```
to install matching dependencies.

Protobuf and gRPC files are **not** included.
The script to generate those files is
```shell
source .venv/bin/activate
chmod +x update-grpc.sh
./update-grpc.sh
```

Now run `main.py` script to rock.
