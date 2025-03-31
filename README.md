# `metriq-gym`

[![Unitary Foundation](https://img.shields.io/badge/Supported%20By-Unitary%20Foundation-FFFF00.svg)](https://unitary.foundation)
[![Discord Chat](https://img.shields.io/badge/dynamic/json?color=blue&label=Discord&query=approximate_presence_count&suffix=%20online.&url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2FJqVGmpkP96%3Fwith_counts%3Dtrue)](http://discord.unitary.foundation)


`metriq-gym` is a Python framework for implementing and running standard quantum benchmarks on different quantum devices by different providers.

- _Open_ – Open-source since its inception and fully developed in public.
- _Transparent_ – All benchmark parameters are defined in a schema file and the benchmark code is reviewable by the community.
- _Cross-platform_ – Supports running benchmarks on multiple quantum hardware providers (_integration powered by [qBraid-SDK](https://github.com/qBraid/qBraid)_)
- _User-friendly_ – Provides a simple command-line interface for dispatching, monitoring, and polling benchmark jobs (you can go on with your life while your job waits in the queue).

Data generated by metriq-gym is intended to be published on https://metriq.info.

**Join the conversation!** 
- For code, repo, or theory questions, especially those requiring more detailed responses, submit a [Discussion](https://github.com/unitaryfund/metriq-gym/discussions).
- For casual or time sensitive questions, chat with us on [Discord](http://discord.unitary.foundation)'s `#metriq` channel.


## Setup

You will require Python 3.12 or above, and [`poetry`](https://python-poetry.org/).

### Cloning the repo
When cloning the metriq-gym repository use:

```sh
git clone --recurse-submodules https://github.com/unitaryfund/metriq-gym.git
```

This allows you to fetch [qiskit-device-benchmarking](https://github.com/qiskit-community/qiskit-device-benchmarking) as a git submodule
for a set of some of the IBM benchmarks.

### Installation
Once you have `poetry` installed and the repository cloned, run:

```sh
poetry install
```
from the root folder of the project, in order to install the project dependencies. 
We recommend doing this in an isolated virtual environment. See [Poetry documentation](https://python-poetry.org/docs/managing-environments/) for more information on managing virtual environments.

If you use `pyenv`, here is a quick start guide to set up the environment and install all dependencies:

```sh
pyenv install 3.13  
pyenv local 3.13 
poetry install
eval $(poetry env activate)
```
All Python commands below should be run in the virtual environment.

## Running benchmarks

### Credential management

To run on hardware, each hardware provider offers API tokens that are required to interact with their quantum devices.
In order to run on these devices, you will need to follow the instructions on the respective pages of the providers and
obtain API keys from them.

The `.env.example` file illustrates how to specify the API keys once you have acquired them. You will need to create a
`.env` file in the same directory as `.env.example` and populate the values of these variables accordingly.

### Workflow

You can dispatch a job by specifying the parameters of the job you wish to launch in a configuration file. 

```sh
python metriq_gym/run.py dispatch <BENCHMARK_JSON> --provider <PROVIDER> --device <DEVICE>
```

Refer to the `schemas/` directory for example schema files for other supported benchmarks.


If running on quantum cloud hardware, the job will be added to a polling queue. The status of the queue can be checked with

```sh
python metriq_gym/run.py poll --job_id <METRIQ_GYM_JOB_ID>
```

where `<METRIQ_GYM_JOB_ID>` is the assigned job ID of the job that was dispatched as provided by `metriq-gym`. 

Alternatively, the `poll` action can be used without the `--job_id` flag to view all dispatched jobs, 
and select the one that is of interest.

```sh
python metriq_gym/run.py poll
```

Once your job is complete and you are happy with the results, you can upload it directly to the Metriq web app.

```sh
python metriq_gym/run.py upload --job_id <METRIQ_GYM_JOB_ID> --submission_id <METRIQ_APP_SUBMISSION_ID>
```

### View jobs

You can view all the jobs that have been dispatched by using the `view` action. 
This will display basic information about each job, including its ID, backend, job type, provider, and device.

```sh
python metriq_gym/run.py view
```
In order to view the details of a specific job (e.g., the parameters the job was launched with), 
you can use the `view` action with the `--job_id` flag or select the job by index from the list of all dispatched jobs.

```sh
python metriq_gym/run.py view --job_id <METRIQ_GYM_JOB_ID>
```

### Example: Benchmarking Bell state effective qubits (BSEQ) on IBM hardware
The following example is for IBM, but the general workflow is applicable to any of the supported providers and benchmarks.

To run on IBM hardware, you will also require an IBM token. To obtain this, please visit the [IBM Quantum
Platform](https://quantum.ibm.com/) and include the API token in the local `.env` file as previously described.

The `schemas/examples/` directory houses example JSON configuration files that define the benchmark to run. In this
case, we use the `bseq_example.json` file as we want to run a BSEQ job. The following dispatches a
job on the ibm-sherbrooke device for BSEQ.

```sh
python metriq_gym/run.py dispatch metriq_gym/schemas/examples/bseq.example.json --provider ibm --device ibm_sherbrooke
```

We should see logging information in our terminal to indicate that the dispatch action is taking place:

```sh
INFO - Starting job dispatch...
INFO - Dispatching BSEQ benchmark job on ibm_sherbrooke device...
...
INFO - Job dispatched with ID: 93a06a18-41d8-475a-a030-339fbf3accb9
```

We can confirm that the job has indeed been dispatched and retrieve the associated metriq-gym job ID (along with other pieces of metadata).

```sh
+--------------------------------------+------------+------------------------------------------------------+----------------+----------------------------+
| Metriq-gym Job Id                    | Provider   | Device                                               | Type           | Dispatch time (UTC)        |
+======================================+============+======================================================+================+============================+
| 93a06a18-41d8-475a-a030-339fbf3accb9 | ibm        | ibm_sherbrooke                                       | BSEQ           | 2025-03-05T10:21:18.333703 |
+--------------------------------------+------------+------------------------------------------------------+----------------+----------------------------+
```

We can use the "poll" action to check the status of our job:

```sh
python metriq_gym/run.py poll --job_id 93a06a18-41d8-475a-a030-339fbf3accb9
```

Doing so gives us the results of our job (if it has completed):

```sh
INFO - Polling job...
BSEQResult(largest_connected_size=100, fraction_connected=0.7874015748031497)
```

In the event where the job has not completed, we would receive the following message instead

```sh
INFO - Polling job...
INFO - Job is not yet completed. Please try again later.
```

As a convenience, while we could supply the metriq-gym job ID, we can also poll the job by running `python
metriq_gym/run.py poll` and then selecting the job to poll by index from our local metriq-gym jobs database.

```sh
Available jobs:
+----+--------------------------------------+------------+------------------------------------------------------+----------------+-----------------------------+
|    | Metriq-gym Job Id                    | Provider   | Device                                               | Type           | Dispatch time (UTC)         |
+====+======================================+============+======================================================+================+=============================+
| 0  | 93a06a18-41d8-475a-a030-339fbf3accb9 | ibm        | ibm_sherbrooke                                        | BSEQ           | 2025-03-05T10:21:18.333703 |
+----+--------------------------------------+------------+------------------------------------------------------+----------------+-----------------------------+
Select a job index: 
```

Entering the index (in this case, `0`), polls the same job.

```sh
Select a job index: 0
INFO - Polling job...
```

## Contributing

First, follow the [Setup](#setup) instructions above.

### Updating the submodule
To pull the latest changes from the submodule’s repository:

```sh
cd submodules/qiskit-device-benchmarking
git pull origin main
```

Then, commit the updated submodule reference in your main repository.

### Style guide
We don't have a style guide per se, but we recommend that both linter and formatter 
are run before each commit. In order to guarantee that, please install the pre-commit hook with

```sh
poetry run pre-commit install
```
immediately upon cloning the repository.

### Tests
The suite of unit tests can be run with
```sh
poetry run pytest
```

### Type checking
The project uses [mypy](https://mypy.readthedocs.io/en/stable/) for static type checking. To run mypy, use the following command:
```sh
poetry run mypy
```

### Documentation
The project uses [Sphinx](https://www.sphinx-doc.org/en/master/) to generate documentation. To build the HTML
documentation:

1.Navigate to the docs/ directory:
```sh
cd docs/
```

Run the following command to build the HTML files:
```sh
make html
```

Open the generated `index.html` file located in the `_build/html/` directory to view the documentation.

## Data

[First hardware results are here.](https://github.com/unitaryfund/metriq-gym/wiki/First-Hardware-Data)
