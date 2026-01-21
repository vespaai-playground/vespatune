from argparse import ArgumentParser

from ..splitter import split_data
from . import BaseCommand


def splitter_vespatune_command_factory(args):
    return SplitterVespaTuneCommand(
        args.input_filename,
        args.output_dir,
        args.num_folds,
        args.targets,
        args.task,
        args.seed,
    )


class SplitterVespaTuneCommand(BaseCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        _parser = parser.add_parser(
            "splitter", help="Split data into k folds for cross-validation"
        )
        _parser.add_argument(
            "--input_filename",
            help="Path to input CSV file",
            required=True,
            type=str,
        )
        _parser.add_argument(
            "--output_dir",
            help="Directory to save fold files",
            required=True,
            type=str,
        )
        _parser.add_argument(
            "--num_folds",
            help="Number of folds (default: 5)",
            required=False,
            type=int,
            default=5,
        )
        _parser.add_argument(
            "--targets",
            help="Target column(s) for stratification, separated by ';'",
            required=False,
            type=str,
            default="target",
        )
        _parser.add_argument(
            "--task",
            help="Task type for stratification strategy",
            required=False,
            type=str,
            default=None,
            choices=["classification", "regression"],
        )
        _parser.add_argument(
            "--seed",
            help="Random seed (default: 42)",
            required=False,
            type=int,
            default=42,
        )

        _parser.set_defaults(func=splitter_vespatune_command_factory)

    def __init__(self, input_filename, output_dir, num_folds, targets, task, seed):
        self.input_filename = input_filename
        self.output_dir = output_dir
        self.num_folds = num_folds
        self.targets = targets.split(";") if targets else None
        self.task = task
        self.seed = seed

    def execute(self):
        split_data(
            input_filename=self.input_filename,
            output_dir=self.output_dir,
            num_folds=self.num_folds,
            targets=self.targets,
            task=self.task,
            seed=self.seed,
        )
