from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qbraid.runtime.result_data import MeasCount, GateModelResultData


def flatten_counts(result_data: list["GateModelResultData"]) -> list["MeasCount"]:
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


def serialize_raw_counts(result_data: list["GateModelResultData"]) -> list[dict[str, Any]]:
    """Serialize raw measurement counts from GateModelResultData for JSON export.

    This preserves the structure of the result data for debugging and replay purposes.
    Each element in the returned list corresponds to one GateModelResultData object,
    containing its measurement counts (which may be a single dict or a list of dicts).

    Args:
        result_data: List of GateModelResultData objects from quantum job results.

    Returns:
        List of serializable dictionaries, each with a 'measurement_counts' key.
    """
    serialized: list[dict[str, Any]] = []
    for result in result_data:
        counts = result.measurement_counts
        if isinstance(counts, list):
            # Convert each MeasCount to a plain dict
            serialized.append({"measurement_counts": [dict(c) for c in counts]})
        elif counts is not None:
            serialized.append({"measurement_counts": dict(counts)})
        else:
            serialized.append({"measurement_counts": None})
    return serialized
