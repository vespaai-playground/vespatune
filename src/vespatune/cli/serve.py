import os
from argparse import ArgumentParser

import uvicorn

from . import BaseCommand


def serve_vespatune_command_factory(args):
    return ServeVespaTuneCommand(args.model_path, args.port, args.host, args.workers, args.reload)


class ServeVespaTuneCommand(BaseCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        _parser = parser.add_parser("serve", help="Serve VespaTune API using ONNX model")
        _parser.add_argument(
            "--model_path",
            help="Path to ONNX export directory (created by 'vespatune export')",
            required=False,
            type=str,
        )
        _parser.add_argument("--port", help="Port to serve on", default=9999, type=int, required=False)
        _parser.add_argument(
            "--host",
            help="Host to serve on",
            default="127.0.0.1",
            type=str,
            required=False,
        )
        _parser.add_argument("--workers", help="Number of workers", default=1, type=int, required=False)
        _parser.add_argument(
            "--reload",
            help="Enable auto-reload for development",
            action="store_true",
            required=False,
        )
        _parser.set_defaults(func=serve_vespatune_command_factory)

    def __init__(self, model_path, port, host, workers, reload):
        self.model_path = model_path
        self.port = port
        self.host = host
        self.workers = workers
        self.reload = reload

    def execute(self):
        if self.model_path:
            os.environ["VESPATUNE_MODEL_PATH"] = self.model_path
        # run app using uvicorn
        uvicorn.run(
            "vespatune.api:app",
            host=self.host,
            port=self.port,
            reload=self.reload,
            workers=self.workers if not self.reload else 1,
        )
