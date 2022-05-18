from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Counts:
    correct: int = 0
    possible: int = 0
    predicted: int = 0

    def __add__(self, other: Counts) -> Counts:
        return Counts(
            correct=self.correct + other.correct,
            possible=self.possible + other.possible,
            predicted=self.predicted + other.predicted,
        )


@dataclass
class DetailedCounts:
    overall: Counts = Counts()
    small: Counts = Counts()
    medium: Counts = Counts()
    large: Counts = Counts()
    max_1: Counts = Counts()
    max_10: Counts = Counts()
    max_100: Counts = Counts()

    def __add__(self, other: DetailedCounts) -> DetailedCounts:
        return DetailedCounts(
            overall=self.overall + other.overall,
            small=self.small + other.small,
            medium=self.medium + other.medium,
            large=self.large + other.large,
            max_1=self.max_1 + other.max_1,
            max_10=self.max_10 + other.max_10,
            max_100=self.max_100 + other.max_100,
        )
