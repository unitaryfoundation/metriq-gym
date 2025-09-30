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
* `Poetry <https://python-poetry.org/docs/#installation>`_ for managing dependencies

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

Install dependencies into the Poetry-managed virtual environment:

.. code-block:: sh

   poetry install

We recommend working inside an isolated environment. See `Poetry documentation
<https://python-poetry.org/docs/managing-environments/>`_ for tips on managing virtual environments.

For ``pyenv`` users, the following commands configure Python and install dependencies:

.. code-block:: sh

   pyenv install 3.13
   pyenv local 3.13
   poetry install
   eval $(poetry env activate)

Run all subsequent Python commands inside the activated environment.

Contributing
------------

Complete the setup steps above before making changes.

Style Guide
-----------

Run the linter and formatter before each commit. Install the pre-commit hooks right after cloning:

.. code-block:: sh

   poetry run pre-commit install

Tests
-----

The entire suite of tests can be run with

.. code-block:: sh

   poetry run pytest

Unit tests only can be run with

.. code-block:: sh

   poetry run pytest -m "not e2e"

End-to-end tests only can be run with

.. code-block:: sh

   poetry run pytest -m e2e

Type Checking
-------------

The project uses `mypy <https://mypy.readthedocs.io/en/stable/>`_ for static analysis. To run mypy, use the following
command:

.. code-block:: sh

   poetry run mypy

Documentation
-------------

The project uses `Sphinx <https://www.sphinx-doc.org/en/master/>`_ to generate documentation. Build the HTML files from
inside the ``docs`` directory:

.. code-block:: sh

   cd docs
   poetry run make html

Open ``_build/html/index.html`` to view the render locally.

Contribution Checklist
----------------------

- Follow the `Conventional Commit <https://www.conventionalcommits.org/en/v1.0.0/>`_ style used in the history (for
  example, ``fix: align quantinuum topology check``).
- Rebase onto the latest ``main`` before opening a pull request.
- Link issues or discussions and attach CLI output or screenshots for user-facing changes.
- Coordinate dependency or submodule updates with maintainers when in doubt.
