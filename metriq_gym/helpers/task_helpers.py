from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qbraid.runtime.result_data import MeasCount, GateModelResultData


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
        ValueError: If any result carries no usable measurement counts. A result is not
            silently dropped, since doing so lets a benchmark fall through to reporting a
            fabricated zero-initialized score as though it had actually been measured.
    """
    flat_counts: list[MeasCount] = []
    for result in result_data:
        if isinstance(result.measurement_counts, list):
            flat_counts.extend(result.measurement_counts)
        elif result.measurement_counts is not None:
            flat_counts.append(result.measurement_counts)
        else:
            raise ValueError(
                f"Cannot extract measurement counts: {_describe_missing_counts(result)}."
            )
    return flat_counts
