from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qbraid.runtime.result_data import GateModelResultData, MeasCount


def _describe_missing_counts(result: "GateModelResultData") -> str:
    """Explain why a result has no usable measurement counts, for the error raised by
    flatten_counts. Distinguishes "provider returned probabilities instead of counts"
    (e.g. OriginQ simulators) from "no measurement data at all".
    """
    try:
        result.get_probabilities()
    except ValueError:
        return "no measurement counts or probabilities are available for this result"
    return (
        "the provider returned measurement probabilities instead of sampled counts "
        "(measurement_counts is None, but probabilities are available). Synthesizing "
        "sampled counts from probabilities is not supported, so this result cannot be scored"
    )


def flatten_counts(result_data: list["GateModelResultData"]) -> list["MeasCount"]:
    """Flatten the measurement counts from a list of GateModelResultData objects.

    This is to seamlessly handle the different ways batching is handled on the provider side.

    Example: if we dispatch a job with 2 circuits, IBM returns one result with a list of 2 MeasCount objects.
    If we dispatch the same job to AWS/Rigetti, we get 2 results each with a single MeasCount object.

    Raises:
        ValueError: If any result carries no usable measurement counts -- this includes
            `None`, an empty dict, an empty list, or a list containing an empty dict. A
            result is not silently dropped or contributes zero counts unnoticed, since
            doing so lets a benchmark fall through to reporting a fabricated
            zero-initialized score as though it had actually been measured.
    """
    flat_counts: list[MeasCount] = []
    for result in result_data:
        counts = result.measurement_counts
        if isinstance(counts, list):
            if not counts or any(not item for item in counts):
                raise ValueError(
                    f"Cannot extract measurement counts: {_describe_missing_counts(result)}."
                )
            flat_counts.extend(counts)
        elif counts:
            flat_counts.append(counts)
        else:
            raise ValueError(
                f"Cannot extract measurement counts: {_describe_missing_counts(result)}."
            )
    return flat_counts
