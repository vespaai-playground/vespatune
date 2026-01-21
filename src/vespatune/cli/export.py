from argparse import ArgumentParser

from ..export import export_model
from . import BaseCommand


def export_vespatune_command_factory(args):
    return ExportVespaTuneCommand(args.model_path, args.output_dir, args.verify)


class ExportVespaTuneCommand(BaseCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        _parser = parser.add_parser("export", help="Export trained model to ONNX format")
        _parser.add_argument(
            "--model_path",
            help="Path to trained model directory",
            required=True,
            type=str,
        )
        _parser.add_argument(
            "--output_dir",
            help="Output directory for ONNX model (default: model_path/onnx)",
            required=False,
            type=str,
            default=None,
        )
        _parser.add_argument(
            "--verify",
            help="Verify exported model using onnxruntime",
            action="store_true",
            required=False,
        )
        _parser.set_defaults(func=export_vespatune_command_factory)

    def __init__(self, model_path, output_dir, verify):
        self.model_path = model_path
        self.output_dir = output_dir
        self.verify = verify

    def execute(self):
        export_model(
            model_path=self.model_path,
            output_dir=self.output_dir,
            verify=self.verify,
        )
