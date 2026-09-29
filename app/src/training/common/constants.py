"""Protected token ids for the byte-level BPE tokenizer in bpe.py.

Ids 0-255 are raw bytes. The protected ids follow them contiguously and come before
the merges, so they are fixed regardless of num_merges, and encode never produces
them. bpe.py starts the merges after the last one, so adding an id here shifts
every merge id and every saved tokenizer has to be retrained.
"""

PAD_ID = 256
MASK_ID = 257  # masked-language-modelling target
CLS_ID = 258  # sequence summary position
SEP_ID = 259  # boundary between two segments
PROTECTED_IDS = [PAD_ID, MASK_ID, CLS_ID, SEP_ID]
