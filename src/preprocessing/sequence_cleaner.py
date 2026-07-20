"""
Module
------
Protein Sequence Cleaning

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
Removes ambiguous amino acids and validates sequence quality.
"""

from typing import Final

INVALID_RESIDUES: Final = {

    "B",

    "J",

    "O",

    "U",

    "X",

    "Z",

}


class SequenceCleaner:
    """
    Cleans raw protein sequences.
    """

    def clean(
        self,
        sequence: str,
    ) -> str:
        """
        Clean sequence.

        Args:
            sequence:
                Raw sequence.

        Returns:
            Cleaned sequence.
        """

        sequence = sequence.upper()

        filtered = [

            residue

            for residue in sequence

            if residue not in INVALID_RESIDUES

        ]

        return "".join(filtered)

    def is_valid(
        self,
        sequence: str,
        threshold: float = 0.10,
    ) -> bool:
        """
        Validate ambiguity ratio.

        Args:
            sequence:
                Raw sequence.

            threshold:
                Maximum ambiguity ratio.

        Returns:
            Validation result.
        """

        if not sequence:

            return False

        invalid = sum(

            residue in INVALID_RESIDUES

            for residue in sequence

        )

        return (invalid / len(sequence)) <= threshold