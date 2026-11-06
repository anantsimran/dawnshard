from training.ablations.multi30k_translator_ablation import (
    ABLATION_REFERENCE,
    BEST_CONFIG,
    COMPONENT_ARMS,
    SCALE_ARMS,
    STUDIES,
)


def test_dropout_arm_sets_every_rate():
    model = COMPONENT_ARMS[1].model
    rates = {model.src.dropout, model.tgt.dropout, model.encoder.dropout, model.decoder.dropout}
    assert rates == {0.0}


def test_block_arms_change_both_stacks():
    for arm in (COMPONENT_ARMS[0], COMPONENT_ARMS[2]):
        assert arm.model.encoder == arm.model.decoder
        assert arm.model.encoder != ABLATION_REFERENCE.model.encoder


def test_studies_run_for_their_own_epochs():
    assert {config.fit.num_epochs for config in STUDIES["ablation"]} == {30}
    assert {config.fit.num_epochs for config in SCALE_ARMS} == {25}
    assert BEST_CONFIG.scheduler is not None
    assert BEST_CONFIG.scheduler.name == "warmup_cosine"
