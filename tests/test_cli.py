import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from vespatune import __version__
from vespatune.cli.export import ExportVespaTuneCommand
from vespatune.cli.predict import PredictVespaTuneCommand
from vespatune.cli.splitter import SplitterVespaTuneCommand
from vespatune.cli.train import TrainVespaTuneCommand
from vespatune.cli.vespatune import main


class TestCLIVersion:
    """Test CLI version display."""

    def test_version_flag(self):
        """Test --version flag."""
        result = subprocess.run(
            [sys.executable, "-m", "vespatune.cli.vespatune", "--version"],
            capture_output=True,
            text=True,
        )
        assert __version__ in result.stdout

    def test_version_short_flag(self):
        """Test -v flag."""
        result = subprocess.run(
            [sys.executable, "-m", "vespatune.cli.vespatune", "-v"],
            capture_output=True,
            text=True,
        )
        assert __version__ in result.stdout


class TestCLIHelp:
    """Test CLI help messages."""

    def test_main_help(self):
        """Test main help message."""
        result = subprocess.run(
            [sys.executable, "-m", "vespatune.cli.vespatune", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "vespatune" in result.stdout
        assert "train" in result.stdout
        assert "predict" in result.stdout
        assert "splitter" in result.stdout
        assert "export" in result.stdout
        assert "serve" in result.stdout

    def test_train_help(self):
        """Test train subcommand help."""
        result = subprocess.run(
            [sys.executable, "-m", "vespatune.cli.vespatune", "train", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--train_filename" in result.stdout
        assert "--valid_filename" in result.stdout
        assert "optional" in result.stdout.lower()  # valid_filename is now optional
        assert "--output" in result.stdout
        assert "--model" in result.stdout

    def test_predict_help(self):
        """Test predict subcommand help."""
        result = subprocess.run(
            [sys.executable, "-m", "vespatune.cli.vespatune", "predict", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--model_path" in result.stdout
        assert "--test_filename" in result.stdout
        assert "--out_filename" in result.stdout

    def test_splitter_help(self):
        """Test splitter subcommand help."""
        result = subprocess.run(
            [sys.executable, "-m", "vespatune.cli.vespatune", "splitter", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--input_filename" in result.stdout
        assert "--output_dir" in result.stdout
        assert "--num_folds" in result.stdout

    def test_export_help(self):
        """Test export subcommand help."""
        result = subprocess.run(
            [sys.executable, "-m", "vespatune.cli.vespatune", "export", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--model_path" in result.stdout
        assert "--output_dir" in result.stdout
        assert "--verify" in result.stdout


class TestSplitterCommand:
    """Test splitter CLI command."""

    def test_splitter_command_init(self):
        """Test SplitterVespaTuneCommand initialization."""
        cmd = SplitterVespaTuneCommand(
            input_filename="data.csv",
            output_dir="/tmp/output",
            num_folds=5,
            targets="target",
            task="classification",
            seed=42,
        )
        assert cmd.input_filename == "data.csv"
        assert cmd.output_dir == "/tmp/output"
        assert cmd.num_folds == 5
        assert cmd.targets == ["target"]
        assert cmd.task == "classification"
        assert cmd.seed == 42

    def test_splitter_command_multiple_targets(self):
        """Test SplitterVespaTuneCommand with multiple targets."""
        cmd = SplitterVespaTuneCommand(
            input_filename="data.csv",
            output_dir="/tmp/output",
            num_folds=3,
            targets="target1;target2",
            task="regression",
            seed=123,
        )
        assert cmd.targets == ["target1", "target2"]

    def test_splitter_command_execute(self, full_data_file):
        """Test splitter command execution."""
        output_dir = os.path.join(full_data_file["temp_dir"], "folds")
        cmd = SplitterVespaTuneCommand(
            input_filename=full_data_file["data_path"],
            output_dir=output_dir,
            num_folds=3,
            targets="target",
            task="classification",
            seed=42,
        )
        cmd.execute()

        # Check that fold files were created
        for i in range(3):
            assert os.path.exists(os.path.join(output_dir, f"train_fold_{i}.csv"))
            assert os.path.exists(os.path.join(output_dir, f"valid_fold_{i}.csv"))


class TestTrainCommand:
    """Test train CLI command."""

    def test_train_command_init(self):
        """Test TrainVespaTuneCommand initialization."""
        cmd = TrainVespaTuneCommand(
            train_filename="train.csv",
            valid_filename="valid.csv",
            idx="id",
            targets="target",
            task="classification",
            output="/tmp/output",
            features=None,
            use_gpu=False,
            seed=42,
            test_filename=None,
            num_trials=10,
            time_limit=60,
            model_type="xgboost",
        )
        assert cmd.train_filename == "train.csv"
        assert cmd.valid_filename == "valid.csv"
        assert cmd.idx == "id"
        assert cmd.targets == ["target"]
        assert cmd.task == "classification"
        assert cmd.output == "/tmp/output"
        assert cmd.features is None
        assert cmd.use_gpu is False
        assert cmd.seed == 42
        assert cmd.num_trials == 10
        assert cmd.time_limit == 60
        assert cmd.model_type == "xgboost"

    def test_train_command_multiple_targets(self):
        """Test TrainVespaTuneCommand with multiple targets."""
        cmd = TrainVespaTuneCommand(
            train_filename="train.csv",
            valid_filename="valid.csv",
            idx="id",
            targets="target1;target2",
            task="regression",
            output="/tmp/output",
            features="feat1;feat2;feat3",
            use_gpu=True,
            seed=123,
            test_filename="test.csv",
            num_trials=50,
            time_limit=300,
            model_type="lightgbm",
        )
        assert cmd.targets == ["target1", "target2"]
        assert cmd.features == ["feat1", "feat2", "feat3"]
        assert cmd.use_gpu is True
        assert cmd.model_type == "lightgbm"

    @patch("vespatune.cli.train.VespaTune")
    def test_train_command_execute(self, mock_vespatune):
        """Test train command execution calls VespaTune correctly."""
        mock_instance = MagicMock()
        mock_vespatune.return_value = mock_instance

        cmd = TrainVespaTuneCommand(
            train_filename="train.csv",
            valid_filename="valid.csv",
            idx="id",
            targets="target",
            task="classification",
            output="/tmp/output",
            features=None,
            use_gpu=False,
            seed=42,
            test_filename=None,
            num_trials=10,
            time_limit=60,
            model_type="xgboost",
        )
        cmd.execute()

        mock_vespatune.assert_called_once_with(
            train_filename="train.csv",
            valid_filename="valid.csv",
            idx="id",
            targets=["target"],
            task="classification",
            output="/tmp/output",
            features=None,
            use_gpu=False,
            seed=42,
            test_filename=None,
            num_trials=10,
            time_limit=60,
            model_type="xgboost",
        )
        mock_instance.train.assert_called_once()

    @patch("vespatune.cli.train.VespaTune")
    def test_train_command_execute_no_validation(self, mock_vespatune):
        """Test train command execution without validation file (auto-split)."""
        mock_instance = MagicMock()
        mock_vespatune.return_value = mock_instance

        cmd = TrainVespaTuneCommand(
            train_filename="train.csv",
            valid_filename=None,
            idx="id",
            targets="target",
            task="classification",
            output="/tmp/output",
            features=None,
            use_gpu=False,
            seed=42,
            test_filename=None,
            num_trials=10,
            time_limit=60,
            model_type="xgboost",
        )
        cmd.execute()

        mock_vespatune.assert_called_once_with(
            train_filename="train.csv",
            valid_filename=None,
            idx="id",
            targets=["target"],
            task="classification",
            output="/tmp/output",
            features=None,
            use_gpu=False,
            seed=42,
            test_filename=None,
            num_trials=10,
            time_limit=60,
            model_type="xgboost",
        )
        mock_instance.train.assert_called_once()


class TestPredictCommand:
    """Test predict CLI command."""

    def test_predict_command_init(self):
        """Test PredictVespaTuneCommand initialization."""
        cmd = PredictVespaTuneCommand(
            model_path="/tmp/model",
            test_filename="test.csv",
            out_filename="predictions.csv",
        )
        assert cmd.model_path == "/tmp/model"
        assert cmd.test_filename == "test.csv"
        assert cmd.out_filename == "predictions.csv"

    @patch("vespatune.cli.predict.VespaTunePredict")
    def test_predict_command_execute(self, mock_predict):
        """Test predict command execution calls VespaTunePredict correctly."""
        mock_instance = MagicMock()
        mock_predict.return_value = mock_instance

        cmd = PredictVespaTuneCommand(
            model_path="/tmp/model",
            test_filename="test.csv",
            out_filename="predictions.csv",
        )
        cmd.execute()

        mock_predict.assert_called_once_with(model_path="/tmp/model")
        mock_instance.predict_file.assert_called_once_with(
            test_filename="test.csv",
            out_filename="predictions.csv",
        )


class TestExportCommand:
    """Test export CLI command."""

    def test_export_command_init(self):
        """Test ExportVespaTuneCommand initialization."""
        cmd = ExportVespaTuneCommand(
            model_path="/tmp/model",
            output_dir="/tmp/onnx",
            verify=True,
        )
        assert cmd.model_path == "/tmp/model"
        assert cmd.output_dir == "/tmp/onnx"
        assert cmd.verify is True

    def test_export_command_init_defaults(self):
        """Test ExportVespaTuneCommand with default values."""
        cmd = ExportVespaTuneCommand(
            model_path="/tmp/model",
            output_dir=None,
            verify=False,
        )
        assert cmd.model_path == "/tmp/model"
        assert cmd.output_dir is None
        assert cmd.verify is False

    @patch("vespatune.cli.export.export_model")
    def test_export_command_execute(self, mock_export):
        """Test export command execution calls export_model correctly."""
        cmd = ExportVespaTuneCommand(
            model_path="/tmp/model",
            output_dir="/tmp/onnx",
            verify=True,
        )
        cmd.execute()

        mock_export.assert_called_once_with(
            model_path="/tmp/model",
            output_dir="/tmp/onnx",
            verify=True,
        )


class TestMainEntryPoint:
    """Test main CLI entry point."""

    @patch("vespatune.cli.vespatune.ServeVespaTuneCommand")
    def test_default_runs_serve(self, mock_serve):
        """Test that running without subcommand starts serve."""
        mock_instance = MagicMock()
        mock_serve.return_value = mock_instance

        with patch("sys.argv", ["vespatune"]):
            main()

        mock_serve.assert_called_once()
        mock_instance.execute.assert_called_once()

    def test_version_exits(self):
        """Test --version flag exits cleanly."""
        with patch("sys.argv", ["vespatune", "--version"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
