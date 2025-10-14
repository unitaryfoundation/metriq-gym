Developer Guide
===============

Follow this guide to work on Metriq-Gym locally and contribute changes. For an overview of the
contribution workflow and review expectations, read the `CONTRIBUTING guide
<https://github.com/unitaryfoundation/metriq-gym/blob/main/CONTRIBUTING.md>`__.

These instructions are for setting up a development environment if you plan to contribute to ``metriq-gym`` or run the
latest version from source.

Prerequisites
-------------

Before you begin, ensure you have the following installed:

* `Python <https://www.python.org/downloads/>`_ (version 3.12 or newer)
* `uv <https://docs.astral.sh/uv/getting-started/installation/>`_ for managing dependencies

Cloning the Repository
----------------------

Clone the repository with submodules and move into the project directory:

.. code-block:: sh

   git clone --recurse-submodules https://github.com/unitaryfoundation/metriq-gym.git
   cd metriq-gym

If you already have a local clone (with or without submodules), keep it current with:

.. code-block:: sh

   git pull --recurse-submodules

Installation
------------

Install dependencies into the uv-managed virtual environment:

.. code-block:: sh

   uv sync --all-groups

``uv`` reads the Python requirement from ``pyproject.toml`` and provisions a compatible interpreter automatically.
After ``uv sync`` the project environment lives in ``.venv``; activate it with ``source .venv/bin/activate`` if you
prefer a shell-based workflow, or continue using ``uv run`` for isolated commands.

.. note::
   macOS users installing optional ``pyqpanda3`` support must also install ``libidn2`` via Homebrew, e.g.

   .. code-block:: sh

      brew reinstall libidn2

   Install the library before invoking ``uv sync`` to avoid missing-symbol errors during the ``pyqpanda3`` build.

Contributing
------------

Complete the setup steps above before making changes.

Style Guide
-----------

Run the linter and formatter before each commit. Install the pre-commit hooks right after cloning:

.. code-block:: sh

   uv run pre-commit install

Tests
-----

The entire suite of tests can be run with

.. code-block:: sh

   uv run pytest

Unit tests only can be run with

.. code-block:: sh

   uv run pytest -m "not e2e"

End-to-end tests only can be run with

.. code-block:: sh

   uv run pytest -m e2e

Type Checking
-------------

The project uses `mypy <https://mypy.readthedocs.io/en/stable/>`_ for static analysis. To run mypy, use the following
command:

.. code-block:: sh

   uv run mypy

Documentation
-------------

The project uses `Sphinx <https://www.sphinx-doc.org/en/master/>`_ to generate documentation. Build the HTML files from
inside the ``docs`` directory:

.. code-block:: sh

   cd docs
   uv run make html

Open ``_build/html/index.html`` to view the render locally.

Contribution Checklist
----------------------

- Follow the `Conventional Commit <https://www.conventionalcommits.org/en/v1.0.0/>`_ style used in the history (for
  example, ``fix: align quantinuum topology check``).
- Rebase onto the latest ``main`` before opening a pull request.
- Link issues or discussions and attach CLI output or screenshots for user-facing changes.
- Coordinate dependency or submodule updates with maintainers when in doubt.
