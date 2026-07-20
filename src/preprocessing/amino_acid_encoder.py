"""
Module
------
Protein Sequence Integer Encoder

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
Converts cleaned amino acid sequences into integer encoded vectors.

Usage
-----
The encoder maps the twenty standard amino acids to integer identifiers.
Padding is represented by zero.
"""

from typing import List

from src.utils.constants import (
    AMINO_ACID_TO_INDEX,
    PADDING_INDEX,
)


class AminoAcidEncoder:
    """
    Converts amino acid sequences into integer indices.
    """

    def encode(
        self,
        sequence: str,
    ) -> List[int]:
        """
        Encode a protein sequence.

        Args:
            sequence:
                Clean protein sequence.

        Returns:
            Integer encoded representation.
        """

        encoded: List[int] = []

        for residue in sequence:

            encoded.append(

                AMINO_ACID_TO_INDEX.get(

                    residue,

                    PADDING_INDEX,

                )

            )

        return encoded