def edit_distance(reference, prediction):
    previous = list(range(len(prediction) + 1))
    for row, reference_char in enumerate(reference, start=1):
        current = [row]
        for column, prediction_char in enumerate(prediction, start=1):
            substitution_cost = reference_char != prediction_char
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(references, predictions):
    total_characters = sum(len(reference) for reference in references)
    if total_characters == 0:
        return 0.0
    total_errors = sum(
        edit_distance(reference, prediction)
        for reference, prediction in zip(references, predictions)
    )
    return total_errors / total_characters


def decode_targets(targets, idx_to_char):
    return [
        "".join(idx_to_char[int(index)] for index in target)
        for target in targets
    ]

