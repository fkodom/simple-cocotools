from typing import Any, Sequence, Tuple


def default_collate_fn(batch: Sequence[Sequence[Any]]) -> Tuple[Any, ...]:
    return tuple(zip(*batch))
