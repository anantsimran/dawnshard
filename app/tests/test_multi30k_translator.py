import pytest
import torch
from training.common import bpe
from training.common.constants import BOS_ID, EOS_ID, PAD_ID
from training.model.multi30k_translator import (
    Multi30kTranslator,
    load_bundle,
    save_bundle,
    translate,
)
from training.transformer.decoding import BeamSearch, Greedy

SEQ_LEN = 12


@pytest.fixture
def model() -> Multi30kTranslator:
    torch.manual_seed(seed=0)
    translator = Multi30kTranslator(
        src_vocab_size=300,
        tgt_vocab_size=300,
        d_model=32,
        src_seq_length=SEQ_LEN,
        tgt_seq_length=SEQ_LEN,
        dropout=0.1,
        num_encoder_heads=4,
        num_decoder_heads=4,
        encoder_ffn_multiplier=2,
        decoder_ffn_multiplier=2,
        num_encoders=2,
        num_decoders=2,
        qk_norm=True,
    )
    return translator.eval()


@pytest.fixture
def src() -> torch.Tensor:
    torch.manual_seed(seed=1)
    ids = torch.randint(low=0, high=256, size=(3, SEQ_LEN))
    ids[:, 0] = BOS_ID
    ids[1, 8:] = PAD_ID
    return ids


def test_decode_step_matches_teacher_forced_forward(model, src):
    tgt_in = torch.randint(low=0, high=256, size=(3, 5))
    tgt_in[:, 0] = BOS_ID
    full = torch.full(size=(3, SEQ_LEN), fill_value=PAD_ID)
    full[:, :5] = tgt_in
    with torch.no_grad():
        logits = model(torch.stack(tensors=[src, full], dim=1)).view(3, SEQ_LEN, -1)  # noqa: NAR001
        memory, memory_mask = model.encode(src=src)
        step = model.decode_step(tokens=tgt_in, memory=memory, memory_mask=memory_mask, cache=None)
    torch.testing.assert_close(actual=step, expected=logits[:, 4], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("cache_cross", [True, False])
def test_cached_greedy_is_token_identical_to_uncached(model, src, cache_cross):
    kwargs = dict(model=model, src=src, max_len=SEQ_LEN)  # noqa: NAR001
    uncached, _, _ = translate(strategy=Greedy(eos_id=EOS_ID, pad_id=PAD_ID), use_cache=False, **kwargs)
    cached, cache_bytes, _ = translate(
        strategy=Greedy(eos_id=EOS_ID, pad_id=PAD_ID), use_cache=True, cache_cross=cache_cross, **kwargs
    )
    assert cached == uncached
    assert cache_bytes > 0
    assert all(len(ids) == SEQ_LEN for ids in cached)  # noqa: NAR001


def test_cached_beam_is_token_identical_to_uncached(model, src):
    # A reorder that copied the wrong cache rows would make the cached beams diverge.
    kwargs = dict(model=model, src=src, max_len=SEQ_LEN)  # noqa: NAR001
    uncached, _, _ = translate(
        strategy=BeamSearch(beam_size=4, eos_id=EOS_ID, pad_id=PAD_ID), use_cache=False, **kwargs
    )
    cached, _, _ = translate(
        strategy=BeamSearch(beam_size=4, eos_id=EOS_ID, pad_id=PAD_ID), use_cache=True, **kwargs
    )
    assert cached == uncached


def test_beam_of_one_equals_greedy_on_the_model(model, src):
    kwargs = dict(model=model, src=src, max_len=SEQ_LEN, use_cache=True)  # noqa: NAR001
    greedy, _, _ = translate(strategy=Greedy(eos_id=EOS_ID, pad_id=PAD_ID), **kwargs)
    beam, _, _ = translate(strategy=BeamSearch(beam_size=1, eos_id=EOS_ID, pad_id=PAD_ID), **kwargs)
    assert beam == greedy


def test_bundle_round_trips_model_and_merges(model, tmp_path):
    merges, _ = bpe.train(text="low lower lowest newer wider", num_merges=5)
    path = tmp_path / "bundle.pt"
    save_bundle(model=model, src_merges=merges, tgt_merges=merges, path=path)
    loaded, src_merges, tgt_merges, _ = load_bundle(path=path, device=torch.device("cpu"))  # noqa: NAR001
    assert bpe.encode(text="lowest", merges=src_merges) == bpe.encode(text="lowest", merges=merges)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(actual=loaded.state_dict()[name], expected=tensor)
