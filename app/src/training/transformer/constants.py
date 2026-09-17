"""Axis names for jaxtyping shape strings, e.g. f"{B} {L} {D_MODEL}".

Using these instead of bare literals keeps every annotation in the package on the
same vocabulary. See README.md in this package for what each axis means.
"""

B = "B"  # batch
L = "L"  # sequence length
D_MODEL = "d_model"  # embedding width
H = "h"  # attention heads
D_K = "d_k"  # width per head, d_model // h
