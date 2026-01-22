import argparse

from .. import __version__
from .export import ExportVespaTuneCommand
from .predict import PredictVespaTuneCommand
from .serve import ServeVespaTuneCommand
from .splitter import SplitterVespaTuneCommand
from .train import TrainVespaTuneCommand


def main():
    parser = argparse.ArgumentParser(
        "VespaTune CLI",
        usage="vespatune [<command>] [<args>]",
        epilog="For more information about a command, run: `vespatune <command> --help`",
    )
    parser.add_argument("--version", "-v", help="Display VespaTune version", action="store_true")
    parser.add_argument("--port", help="Port to serve on (default mode)", default=9999, type=int)
    parser.add_argument("--host", help="Host to serve on (default mode)", default="127.0.0.1", type=str)

    commands_parser = parser.add_subparsers(dest="command", help="commands")
    SplitterVespaTuneCommand.register_subcommand(commands_parser)
    TrainVespaTuneCommand.register_subcommand(commands_parser)
    PredictVespaTuneCommand.register_subcommand(commands_parser)
    ServeVespaTuneCommand.register_subcommand(commands_parser)
    ExportVespaTuneCommand.register_subcommand(commands_parser)

    args = parser.parse_args()

    if args.version:
        print(__version__)
        exit(0)

    # If no subcommand provided, run the UI server with default settings
    if not hasattr(args, "func") or args.func is None:
        command = ServeVespaTuneCommand(model_path=None, port=args.port, host=args.host, workers=1, reload=False)
        command.execute()
        return

    command = args.func(args)
    command.execute()


if __name__ == "__main__":
    main()
