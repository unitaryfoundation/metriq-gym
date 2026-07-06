import urllib.request

import pytest

from metriq_gym.benchmarks.benchmark import BenchmarkResult
from metriq_gym.exporters.github_pr_exporter import (
    GITHUB_API_TIMEOUT_SECONDS,
    GitHubPRExporter,
)


def test_urlopen_github_api_passes_timeout(metriq_job, monkeypatch):
    exporter = GitHubPRExporter(metriq_job, BenchmarkResult())
    called = {}

    class FakeOpener:
        def open(self, req, timeout=None):
            called["url"] = req.full_url
            called["timeout"] = timeout
            return object()

    monkeypatch.setattr(urllib.request, "build_opener", lambda: FakeOpener())

    req = urllib.request.Request("https://api.github.com/user", method="GET")
    exporter._urlopen_github_api(req)

    assert called == {"url": "https://api.github.com/user", "timeout": GITHUB_API_TIMEOUT_SECONDS}


def test_urlopen_github_api_rejects_non_https(metriq_job):
    exporter = GitHubPRExporter(metriq_job, BenchmarkResult())
    req = urllib.request.Request("http://api.github.com/user", method="GET")
    with pytest.raises(ValueError, match="Refusing to open non-GitHub API URL"):
        exporter._urlopen_github_api(req)


def test_urlopen_github_api_rejects_unexpected_host(metriq_job):
    exporter = GitHubPRExporter(metriq_job, BenchmarkResult())
    req = urllib.request.Request("https://example.com/user", method="GET")
    with pytest.raises(ValueError, match="Refusing to open non-GitHub API URL"):
        exporter._urlopen_github_api(req)
