#!/usr/bin/env bash

source .venv/bin/activate
python -m grpc_tools.protoc -Iprotos --python_out=. --grpc_python_out=. protos/*.proto