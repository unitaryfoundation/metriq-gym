"""Tests for CLI commands using Typer's CliRunner.

These tests verify:
- Command help messages and usage
- Argument parsing and validation
- Option handling (including environment variables)
- Error handling for invalid inputs
"""

from typer.testing import CliRunner
from unittest.mock import patch, MagicMock

from metriq_gym.cli import app


# Disable Rich markup/colors for consistent test output across environments
runner = CliRunner(mix_stderr=False, env={"NO_COLOR": "1", "TERM": "dumb"})


class TestMainApp:
    """Tests for the main mgym app."""

    def test_help_shows_usage(self):
        """Main app --help should show available commands."""
        result = runner.invoke(app, ["--help"], color=False)
        assert result.exit_code == 0
        assert "job" in result.output
        assert "suite" in result.output
        assert "Metriq-Gym CLI" in result.output

    def test_no_args_shows_help(self):
        """Running mgym without args should show help."""
        result = runner.invoke(app, [], color=False)
        assert result.exit_code == 0
        assert "job" in result.output
        assert "suite" in result.output


class TestJobCommands:
    """Tests for job subcommands."""

    def test_job_help(self):
        """mgym job --help should list job commands."""
        result = runner.invoke(app, ["job", "--help"], color=False)
        assert result.exit_code == 0
        assert "dispatch" in result.output
        assert "poll" in result.output
        assert "view" in result.output
        assert "delete" in result.output
        assert "upload" in result.output
        assert "estimate" in result.output

    def test_job_dispatch_help(self):
        """mgym job dispatch --help should show usage."""
        result = runner.invoke(app, ["job", "dispatch", "--help"], color=False)
        assert result.exit_code == 0
        assert "CONFIG" in result.output
        assert "--provider" in result.output
        assert "--device" in result.output

    def test_job_dispatch_missing_config_shows_error(self):
        """mgym job dispatch without config should error."""
        result = runner.invoke(app, ["job", "dispatch"], color=False)
        assert result.exit_code != 0
        # Error may be in stdout or stderr depending on Typer version
        combined_output = (result.output or "") + (result.stderr or "")
        assert "Missing argument" in combined_output or "CONFIG" in combined_output

    @patch("metriq_gym.cli.JobManager")
    @patch("metriq_gym.run.dispatch_job")
    def test_job_dispatch_calls_dispatch_job(self, mock_dispatch, mock_jm, tmp_path):
        """mgym job dispatch should call dispatch_job with correct args."""
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")

        runner.invoke(
            app,
            ["job", "dispatch", str(config_file), "-p", "local", "-d", "aer_simulator"],
        )

        # The command should attempt to call dispatch_job
        # (may fail due to config validation, but the call is made)
        mock_dispatch.assert_called_once()
        args = mock_dispatch.call_args[0][0]
        assert args.config == str(config_file)
        assert args.provider == "local"
        assert args.device == "aer_simulator"

    def test_job_poll_help(self):
        """mgym job poll --help should show usage."""
        result = runner.invoke(app, ["job", "poll", "--help"], color=False)
        assert result.exit_code == 0
        assert "--json" in result.output
        assert "--no-cache" in result.output
        assert "JOB_ID" in result.output

    @patch("metriq_gym.cli.JobManager")
    @patch("metriq_gym.run.poll_job")
    def test_job_poll_latest(self, mock_poll, mock_jm):
        """mgym job poll latest should work."""
        runner.invoke(app, ["job", "poll", "latest"])

        mock_poll.assert_called_once()
        args = mock_poll.call_args[0][0]
        assert args.job_id == "latest"
        assert args.no_cache is False

    @patch("metriq_gym.cli.JobManager")
    @patch("metriq_gym.run.poll_job")
    def test_job_poll_with_json_output(self, mock_poll, mock_jm, tmp_path):
        """mgym job poll --json should set json output path."""
        outfile = tmp_path / "output.json"
        runner.invoke(app, ["job", "poll", "latest", "--json", str(outfile)])

        mock_poll.assert_called_once()
        args = mock_poll.call_args[0][0]
        assert args.json == str(outfile)

    @patch("metriq_gym.cli.JobManager")
    @patch("metriq_gym.run.poll_job")
    def test_job_poll_with_no_cache(self, mock_poll, mock_jm):
        """mgym job poll --no-cache should set no_cache flag."""
        runner.invoke(app, ["job", "poll", "latest", "--no-cache"])

        mock_poll.assert_called_once()
        args = mock_poll.call_args[0][0]
        assert args.no_cache is True

    def test_job_view_help(self):
        """mgym job view --help should show usage."""
        result = runner.invoke(app, ["job", "view", "--help"], color=False)
        assert result.exit_code == 0
        assert "JOB_ID" in result.output

    @patch("metriq_gym.cli.JobManager")
    @patch("metriq_gym.run.view_job")
    def test_job_view_calls_view_job(self, mock_view, mock_jm):
        """mgym job view should call view_job."""
        runner.invoke(app, ["job", "view", "test-job-id"])

        mock_view.assert_called_once()
        args = mock_view.call_args[0][0]
        assert args.job_id == "test-job-id"

    def test_job_delete_help(self):
        """mgym job delete --help should show usage."""
        result = runner.invoke(app, ["job", "delete", "--help"], color=False)
        assert result.exit_code == 0
        assert "JOB_ID" in result.output

    @patch("metriq_gym.cli.JobManager")
    @patch("metriq_gym.run.delete_job")
    def test_job_delete_calls_delete_job(self, mock_delete, mock_jm):
        """mgym job delete should call delete_job."""
        runner.invoke(app, ["job", "delete", "test-job-id"])

        mock_delete.assert_called_once()
        args = mock_delete.call_args[0][0]
        assert args.job_id == "test-job-id"

    def test_job_estimate_help(self):
        """mgym job estimate --help should show usage."""
        result = runner.invoke(app, ["job", "estimate", "--help"], color=False)
        assert result.exit_code == 0
        assert "CONFIG" in result.output
        assert "--provider" in result.output
        assert "--device" in result.output

    def test_job_upload_help(self):
        """mgym job upload --help should show all options."""
        result = runner.invoke(app, ["job", "upload", "--help"], color=False)
        assert result.exit_code == 0
        assert "--repo" in result.output
        assert "--base" in result.output
        assert "--dir" in result.output
        assert "--branch" in result.output
        assert "--title" in result.output
        assert "--body" in result.output
        assert "--commit-message" in result.output
        assert "--clone-dir" in result.output
        assert "--dry-run" in result.output

    @patch("metriq_gym.cli.JobManager")
    @patch("metriq_gym.run.upload_job")
    def test_job_upload_with_dry_run(self, mock_upload, mock_jm):
        """mgym job upload --dry-run should set dry_run flag."""
        runner.invoke(app, ["job", "upload", "latest", "--dry-run"])

        mock_upload.assert_called_once()
        args = mock_upload.call_args[0][0]
        assert args.job_id == "latest"
        assert args.dry_run is True

    @patch("metriq_gym.cli.JobManager")
    @patch("metriq_gym.run.upload_job")
    def test_job_upload_with_all_options(self, mock_upload, mock_jm):
        """mgym job upload with all options should pass them correctly."""
        runner.invoke(
            app,
            [
                "job",
                "upload",
                "test-id",
                "--repo",
                "owner/repo",
                "--base",
                "develop",
                "--dir",
                "data/results",
                "--branch",
                "feature-branch",
                "--title",
                "My PR",
                "--body",
                "PR description",
                "--commit-message",
                "Add results",
                "--clone-dir",
                "/tmp/clone",
                "--dry-run",
            ],
        )

        mock_upload.assert_called_once()
        args = mock_upload.call_args[0][0]
        assert args.job_id == "test-id"
        assert args.repo == "owner/repo"
        assert args.base_branch == "develop"
        assert args.upload_dir == "data/results"
        assert args.branch_name == "feature-branch"
        assert args.pr_title == "My PR"
        assert args.pr_body == "PR description"
        assert args.commit_message == "Add results"
        assert args.clone_dir == "/tmp/clone"
        assert args.dry_run is True

    @patch("metriq_gym.cli.JobManager")
    @patch("metriq_gym.run.upload_job")
    def test_job_upload_default_values(self, mock_upload, mock_jm):
        """mgym job upload should have correct default values."""
        runner.invoke(app, ["job", "upload", "latest"])

        mock_upload.assert_called_once()
        args = mock_upload.call_args[0][0]
        assert args.repo == "unitaryfoundation/metriq-data"
        assert args.base_branch == "main"
        assert args.dry_run is False


class TestSuiteCommands:
    """Tests for suite subcommands."""

    def test_suite_help(self):
        """mgym suite --help should list suite commands."""
        result = runner.invoke(app, ["suite", "--help"], color=False)
        assert result.exit_code == 0
        assert "dispatch" in result.output
        assert "poll" in result.output
        assert "view" in result.output
        assert "delete" in result.output
        assert "upload" in result.output

    def test_suite_dispatch_help(self):
        """mgym suite dispatch --help should show usage."""
        result = runner.invoke(app, ["suite", "dispatch", "--help"], color=False)
        assert result.exit_code == 0
        assert "SUITE_CONFIG" in result.output
        assert "--provider" in result.output
        assert "--device" in result.output

    def test_suite_dispatch_missing_config_shows_error(self):
        """mgym suite dispatch without config should error."""
        result = runner.invoke(app, ["suite", "dispatch"], color=False)
        assert result.exit_code != 0
        # Error may be in stdout or stderr depending on Typer version
        combined_output = (result.output or "") + (result.stderr or "")
        assert "Missing argument" in combined_output or "SUITE_CONFIG" in combined_output

    @patch("metriq_gym.cli.JobManager")
    @patch("metriq_gym.run.dispatch_suite")
    def test_suite_dispatch_calls_dispatch_suite(self, mock_dispatch, mock_jm, tmp_path):
        """mgym suite dispatch should call dispatch_suite with correct args."""
        config_file = tmp_path / "suite.json"
        config_file.write_text("{}")

        runner.invoke(
            app,
            ["suite", "dispatch", str(config_file), "-p", "local", "-d", "aer_simulator"],
        )

        mock_dispatch.assert_called_once()
        args = mock_dispatch.call_args[0][0]
        assert args.suite_config == str(config_file)
        assert args.provider == "local"
        assert args.device == "aer_simulator"

    def test_suite_poll_help(self):
        """mgym suite poll --help should show usage."""
        result = runner.invoke(app, ["suite", "poll", "--help"], color=False)
        assert result.exit_code == 0
        assert "--json" in result.output
        assert "--no-cache" in result.output
        assert "SUITE_ID" in result.output

    @patch("metriq_gym.cli.JobManager")
    @patch("metriq_gym.run.poll_suite")
    def test_suite_poll_calls_poll_suite(self, mock_poll, mock_jm):
        """mgym suite poll should call poll_suite."""
        runner.invoke(app, ["suite", "poll", "test-suite-id"])

        mock_poll.assert_called_once()
        args = mock_poll.call_args[0][0]
        assert args.suite_id == "test-suite-id"

    @patch("metriq_gym.cli.JobManager")
    @patch("metriq_gym.run.poll_suite")
    def test_suite_poll_with_options(self, mock_poll, mock_jm, tmp_path):
        """mgym suite poll with options should pass them correctly."""
        outfile = tmp_path / "suite_output.json"
        runner.invoke(app, ["suite", "poll", "test-suite", "--json", str(outfile), "--no-cache"])

        mock_poll.assert_called_once()
        args = mock_poll.call_args[0][0]
        assert args.suite_id == "test-suite"
        assert args.json == str(outfile)
        assert args.no_cache is True

    def test_suite_view_help(self):
        """mgym suite view --help should show usage."""
        result = runner.invoke(app, ["suite", "view", "--help"], color=False)
        assert result.exit_code == 0
        assert "SUITE_ID" in result.output

    @patch("metriq_gym.cli.JobManager")
    @patch("metriq_gym.run.view_suite")
    def test_suite_view_calls_view_suite(self, mock_view, mock_jm):
        """mgym suite view should call view_suite."""
        runner.invoke(app, ["suite", "view", "test-suite-id"])

        mock_view.assert_called_once()
        args = mock_view.call_args[0][0]
        assert args.suite_id == "test-suite-id"

    def test_suite_delete_help(self):
        """mgym suite delete --help should show usage."""
        result = runner.invoke(app, ["suite", "delete", "--help"], color=False)
        assert result.exit_code == 0
        assert "SUITE_ID" in result.output

    @patch("metriq_gym.cli.JobManager")
    @patch("metriq_gym.run.delete_suite")
    def test_suite_delete_calls_delete_suite(self, mock_delete, mock_jm):
        """mgym suite delete should call delete_suite."""
        runner.invoke(app, ["suite", "delete", "test-suite-id"])

        mock_delete.assert_called_once()
        args = mock_delete.call_args[0][0]
        assert args.suite_id == "test-suite-id"

    def test_suite_upload_help(self):
        """mgym suite upload --help should show all options."""
        result = runner.invoke(app, ["suite", "upload", "--help"], color=False)
        assert result.exit_code == 0
        assert "--repo" in result.output
        assert "--base" in result.output
        assert "--dry-run" in result.output

    @patch("metriq_gym.cli.JobManager")
    @patch("metriq_gym.run.upload_suite")
    def test_suite_upload_with_dry_run(self, mock_upload, mock_jm):
        """mgym suite upload --dry-run should set dry_run flag."""
        runner.invoke(app, ["suite", "upload", "test-suite", "--dry-run"])

        mock_upload.assert_called_once()
        args = mock_upload.call_args[0][0]
        assert args.suite_id == "test-suite"
        assert args.dry_run is True

    @patch("metriq_gym.cli.JobManager")
    @patch("metriq_gym.run.upload_suite")
    def test_suite_upload_default_values(self, mock_upload, mock_jm):
        """mgym suite upload should have correct default values."""
        runner.invoke(app, ["suite", "upload", "test-suite"])

        mock_upload.assert_called_once()
        args = mock_upload.call_args[0][0]
        assert args.repo == "unitaryfoundation/metriq-data"
        assert args.base_branch == "main"
        assert args.dry_run is False


class TestEnvironmentVariables:
    """Tests for environment variable support."""

    @patch("metriq_gym.cli.JobManager")
    @patch("metriq_gym.run.upload_job")
    def test_job_upload_repo_from_env(self, mock_upload, mock_jm, monkeypatch):
        """MGYM_UPLOAD_REPO env var should set repo option."""
        monkeypatch.setenv("MGYM_UPLOAD_REPO", "custom/repo")

        runner.invoke(app, ["job", "upload", "latest"], env={"MGYM_UPLOAD_REPO": "custom/repo"})

        mock_upload.assert_called_once()
        args = mock_upload.call_args[0][0]
        assert args.repo == "custom/repo"

    @patch("metriq_gym.cli.JobManager")
    @patch("metriq_gym.run.upload_job")
    def test_job_upload_base_branch_from_env(self, mock_upload, mock_jm):
        """MGYM_UPLOAD_BASE_BRANCH env var should set base option."""
        runner.invoke(
            app,
            ["job", "upload", "latest"],
            env={"MGYM_UPLOAD_BASE_BRANCH": "develop"},
        )

        mock_upload.assert_called_once()
        args = mock_upload.call_args[0][0]
        assert args.base_branch == "develop"

    @patch("metriq_gym.cli.JobManager")
    @patch("metriq_gym.run.upload_job")
    def test_cli_option_overrides_env_var(self, mock_upload, mock_jm):
        """CLI option should override environment variable."""
        runner.invoke(
            app,
            ["job", "upload", "latest", "--repo", "cli/repo"],
            env={"MGYM_UPLOAD_REPO": "env/repo"},
        )

        mock_upload.assert_called_once()
        args = mock_upload.call_args[0][0]
        assert args.repo == "cli/repo"


class TestPromptForJob:
    """Tests for the prompt_for_job helper function."""

    def test_prompt_for_job_with_latest(self):
        """prompt_for_job with 'latest' should return latest job."""
        from metriq_gym.cli import prompt_for_job

        mock_jm = MagicMock()
        mock_job = MagicMock()
        mock_jm.get_latest_job.return_value = mock_job

        result = prompt_for_job("latest", mock_jm)

        assert result == mock_job
        mock_jm.get_latest_job.assert_called_once()

    def test_prompt_for_job_with_specific_id(self):
        """prompt_for_job with specific ID should return that job."""
        from metriq_gym.cli import prompt_for_job

        mock_jm = MagicMock()
        mock_job = MagicMock()
        mock_jm.get_job.return_value = mock_job

        result = prompt_for_job("specific-id", mock_jm)

        assert result == mock_job
        mock_jm.get_job.assert_called_once_with("specific-id")

    def test_prompt_for_job_no_jobs(self, capsys):
        """prompt_for_job with no jobs should print message and return None."""
        from metriq_gym.cli import prompt_for_job

        mock_jm = MagicMock()
        mock_jm.get_jobs.return_value = []

        result = prompt_for_job(None, mock_jm)

        assert result is None
        captured = capsys.readouterr()
        assert "No jobs found" in captured.out
