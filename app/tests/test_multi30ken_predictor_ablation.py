from training.ablations.multi30ken_predictor_ablation import (
    ABLATION_ARMS,
    BASELINE,
    SWEEP,
    one_at_a_time_configs,
)


def test_one_at_a_time_crosses_every_config_with_constant_and_cosine():
    configs = one_at_a_time_configs(baseline=BASELINE, sweep=SWEEP)
    assert len(configs) == 2 * (1 + len(SWEEP))  # noqa: NAR001
    assert configs[0].scheduler is None
    assert configs[1].scheduler is not None


def test_ablation_arms_run_at_d_model_128_for_100_epochs():
    assert {(arm.model.d_model, arm.fit.num_epochs) for arm in ABLATION_ARMS} == {(128, 100)}
