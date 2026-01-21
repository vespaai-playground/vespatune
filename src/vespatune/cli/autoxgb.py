import argparse

from .. import __version__
from .predict import PredictVespaTuneCommand
from .serve import ServeVespaTuneCommand
from .train import TrainVespaTuneCommand


def main():
    parser = argparse.ArgumentParser(
        "VespaTune CLI",
        usage="vespatune <command> [<args>]",
        epilog="For more information about a command, run: `vespatune <command> --help`",
    )
    parser.add_argument(
        "--version", "-v", help="Display VespaTune version", action="store_true"
    )

    commands_parser = parser.add_subparsers(help="commands")
    TrainVespaTuneCommand.register_subcommand(commands_parser)
    PredictVespaTuneCommand.register_subcommand(commands_parser)
    ServeVespaTuneCommand.register_subcommand(commands_parser)

    args = parser.parse_args()

    if args.version:
        print(__version__)
        exit(0)

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    command = args.func(args)
    command.execute()


if __name__ == "__main__":
    main()
