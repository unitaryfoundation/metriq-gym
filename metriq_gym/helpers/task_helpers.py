from qbraid.runtime.result_data import MeasCount, GateModelResultData


def flatten_counts(result_data: list[GateModelResultData]) -> list[MeasCount]:
    """Flatten the measurement counts from a list of GateModelResultData objects.

    This is to seamlessly handle the different ways batching is handled on the provider side.

    Example: if we dispatch a job with 2 circuits, IBM returns one result with a list of 2 MeasCount objects.
    If we dispatch the same job to AWS/Rigetti, we get 2 results each with a single MeasCount object.
    """
    flat_counts: list[MeasCount] = []
    for result in result_data:
        if isinstance(result.measurement_counts, list):
            flat_counts.extend(result.measurement_counts)
        elif result.measurement_counts is not None:
            flat_counts.append(result.measurement_counts)
    return flat_counts
