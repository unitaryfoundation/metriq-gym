###############################################################################
# (C) Quantum Economic Development Consortium (QED-C) 2021.
# Technical Advisory Committee on Standards and Benchmarks (TAC)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
##########################
# Metrics Module
#
# This module contains methods to initialize, store, aggregate and
# plot metrics collected in the benchmark programs
#
# Metrics are indexed by group id (e.g. circuit size), circuit id (e.g. secret string)
# and metric name.
# Multiple metrics with different names may be stored, each with a single value.
#
# The 'aggregate' method accumulates all circuit-specific metrics for each group
# and creates an average that may be reported and plotted across all groups
#

############################################
# DATA ANALYSIS - FIDELITY CALCULATIONS

import numpy as np

## Uniform distribution function commonly used


def uniform_dist(num_state_qubits):
    dist = {}
    for i in range(2**num_state_qubits):
        key = bin(i)[2:].zfill(num_state_qubits)
        dist[key] = 1 / (2**num_state_qubits)
    return dist


### Analysis methods to be expanded and eventually compiled into a separate analysis.py file


# Compute the fidelity based on Hellinger distance between two discrete probability distributions
def hellinger_fidelity_with_expected(p, q):
    """p: result distribution, may be passed as a counts distribution
        q: the expected distribution to be compared against

    References:
        `Hellinger Distance @ wikipedia <https://en.wikipedia.org/wiki/Hellinger_distance>`_
        Qiskit Hellinger Fidelity Function
    """
    p_sum = sum(p.values())
    q_sum = sum(q.values())

    if q_sum == 0:
        print(
            "ERROR: polarization_fidelity(), expected distribution is invalid, all counts equal to 0"
        )
        return 0

    p_normed = {}
    for key, val in p.items():
        p_normed[key] = val / p_sum

    q_normed = {}
    for key, val in q.items():
        q_normed[key] = val / q_sum

    total = 0
    for key, val in p_normed.items():
        if key in q_normed.keys():
            total += (np.sqrt(val) - np.sqrt(q_normed[key])) ** 2
            del q_normed[key]
        else:
            total += val
    total += sum(q_normed.values())

    # in some situations (error mitigation) this can go negative, use abs value
    if total < 0:
        print("WARNING: using absolute value in fidelity calculation")
        total = abs(total)

    dist = np.sqrt(total) / np.sqrt(2)
    fidelity = (1 - dist**2) ** 2

    return fidelity


def rescale_fidelity(fidelity, floor_fidelity, new_floor_fidelity):
    """
    Linearly rescales our fidelities to allow comparisons of fidelities across benchmarks

    fidelity: raw fidelity to rescale
    floor_fidelity: threshold fidelity which is equivalent to random guessing
    new_floor_fidelity: what we rescale the floor_fidelity to

    Ex, with floor_fidelity = 0.25, new_floor_fidelity = 0.0:
        1 -> 1;
        0.25 -> 0;
        0.5 -> 0.3333;
    """
    rescaled_fidelity = (1 - new_floor_fidelity) / (1 - floor_fidelity) * (fidelity - 1) + 1

    # ensure fidelity is within bounds (0, 1)
    if rescaled_fidelity < 0:
        rescaled_fidelity = 0.0
    if rescaled_fidelity > 1:
        rescaled_fidelity = 1.0

    return rescaled_fidelity


def polarization_fidelity(counts, correct_dist, thermal_dist=None):
    """
    Combines Hellinger fidelity and polarization rescaling into fidelity calculation
    used in every benchmark

    counts: the measurement outcomes after `num_shots` algorithm runs
    correct_dist: the distribution we expect to get for the algorithm running perfectly
    thermal_dist: optional distribution to pass in distribution from a uniform
                  superposition over all states. If `None`: generated as
                  `uniform_dist` with the same qubits as in `counts`

    returns both polarization fidelity and the hellinger fidelity

    Polarization from: `https://arxiv.org/abs/2008.11294v1`
    """

    # get length of random key in correct_dist to find how many qubits measured
    num_measured_qubits = len(list(correct_dist.keys())[0])

    # ensure that all keys in counts are zero padded to this length
    counts = {k.zfill(num_measured_qubits): v for k, v in counts.items()}

    # calculate hellinger fidelity between measured expectation values and correct distribution
    hf_fidelity = hellinger_fidelity_with_expected(counts, correct_dist)

    # to limit cpu and memory utilization, skip noise correction if more than 16 measured qubits
    if num_measured_qubits > 16:
        return {"fidelity": hf_fidelity, "hf_fidelity": hf_fidelity}

    # if not provided, generate thermal dist based on number of qubits
    if thermal_dist is None:
        thermal_dist = uniform_dist(num_measured_qubits)

    # set our fidelity rescaling value as the hellinger fidelity for a depolarized state
    floor_fidelity = hellinger_fidelity_with_expected(thermal_dist, correct_dist)

    # rescale fidelity result so uniform superposition (random guessing) returns fidelity
    # rescaled to 0 to provide a better measure of success of the algorithm (polarization)
    new_floor_fidelity = 0
    fidelity = rescale_fidelity(hf_fidelity, floor_fidelity, new_floor_fidelity)

    return {"fidelity": fidelity, "hf_fidelity": hf_fidelity}
