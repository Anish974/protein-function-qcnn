"""
Module
------
Sequence Padding

Author
------
Amit Pimpalkar

Organization
------------
RBU, Nagpur

Year
----
2026

Purpose
-------
Produces fixed-length protein sequences suitable for CNN processing.
"""

from typing import List


class SequencePadder:
    """
    Pads or truncates encoded sequences.
    """

    def __init__(
        self,
        max_length: int = 300,
    ):

        self.max_length = max_length

    def apply(
        self,
        encoded_sequence: List[int],
    ) -> List[int]:
        """
        Pad or trim sequence.

        Args:
            encoded_sequence:
                Integer encoded sequence.

        Returns:
            Fixed-length sequence.
        """

        if len(encoded_sequence) >= self.max_length:

            return encoded_sequence[: self.max_length]

        padding = [

            0

        ] * (

            self.max_length -

            len(encoded_sequence)

        )

        return encoded_sequence + padding