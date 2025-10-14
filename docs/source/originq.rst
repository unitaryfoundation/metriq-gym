OriginQ Provider Guide
======================

Use this guide to configure the OriginQ (Wukong) integration and run Metriq-Gym
benchmarks on Origin hardware or simulators.

Prerequisites
-------------

* A valid OriginQ QCloud account and API token. Copy the token from the OriginQ
  workbench portal.
* The Metriq-Gym optional dependency ``pyqpanda3`` (installed automatically when
  you run ``uv sync --all-groups`` or ``uv sync`` on this project).
* macOS users must install ``libidn2`` before building ``pyqpanda3``:

  .. code-block:: sh

     brew reinstall libidn2

  Install the library prior to running ``uv sync`` to avoid missing symbol
  errors during the ``pyqpanda3`` build.

Authentication
--------------

Set the Origin API token with the ``ORIGIN_API_KEY`` environment variable (the
CLI and provider helpers look it up on first use). You can export it in your
shell or place it in ``.env`` alongside other provider credentials:

.. code-block:: sh

   export ORIGIN_API_KEY="<your-token>"

Device Selection
----------------

Origin currently exposes a pair of Wukong superconducting devices and several
simulators. Metriq-Gym registers the following identifiers via the qBraid entry
point system:

* ``origin_wukong`` – Alias for the 102-qubit hardware backend (ID
  ``WK_C102_400``). This is the default when examples reference Wukong.
* ``72`` – 72-qubit hardware backend.
* ``full_amplitude`` / ``partial_amplitude`` / ``single_amplitude`` – Simulator
  backends with increasing qubit limits. See
  ``metriq_gym/origin/_constants.py`` for the supported maxima.

To verify which devices your account can access, you can query the provider
catalog directly:

.. code-block:: sh

   uv run python -c "from metriq_gym.run import load_provider; \
provider = load_provider('origin'); \
print([device.id for device in provider.get_devices()])"

Dispatching Benchmarks
----------------------

Dispatch jobs the same way as with other providers, passing the desired Origin
device identifier to ``--device``. A few examples:

.. code-block:: sh

   # 102-qubit hardware (alias)
   uv run mgym job dispatch metriq_gym/schemas/examples/wit.example.json \
     --provider origin --device origin_wukong

   # 72-qubit hardware (numeric ID)
   uv run mgym job dispatch metriq_gym/schemas/examples/wit.example.json \
     --provider origin --device 72

   # Simulator with up to 35 qubits
   uv run mgym job dispatch metriq_gym/schemas/examples/wit.example.json \
     --provider origin --device full_amplitude

If you pass an unsupported device name or one your account lacks permission to
use, OriginQ returns an error such as ``resource is null``. Re-run the provider
listing command above to confirm the available identifiers, then retry with a
supported value.

