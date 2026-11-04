from training.serve.block_manager import BlockManager
from training.serve.scheduler import Request, Scheduler, SchedulerConfig, State


def _scheduler(blocks=64, **config):
    return Scheduler(config=SchedulerConfig(**config), blocks=BlockManager(num_blocks=blocks, block_size=4, prefix_caching=False))


def test_respects_the_token_budget_and_the_sequence_cap():
    scheduler = _scheduler(max_num_seqs=3, max_tokens_per_step=20, chunk_size=8)
    for i in range(5):
        scheduler.add(request=Request(request_id=f"r{i}", prompt=list(range(10)), max_new_tokens=4))
    batch = scheduler.schedule()
    assert sum(n for _, n in batch) <= 20
    assert len(batch) <= 3 and len(scheduler.running) <= 3
    assert all(n <= 8 for _, n in batch)


def test_decodes_are_scheduled_before_new_prefills():
    scheduler = _scheduler(max_tokens_per_step=10, chunk_size=10)
    first = Request(request_id="old", prompt=list(range(4)), max_new_tokens=4)
    scheduler.add(request=first)
    scheduler.schedule()
    first.computed = 4
    first.output.append(1)
    scheduler.add(request=Request(request_id="new", prompt=list(range(20)), max_new_tokens=4))
    batch = scheduler.schedule()
    assert batch[0] == (first, 1)


def test_preemption_frees_blocks_and_requeues_first():
    scheduler = _scheduler(blocks=4, max_tokens_per_step=64, chunk_size=64)
    a = Request(request_id="a", prompt=list(range(8)), max_new_tokens=8)
    b = Request(request_id="b", prompt=list(range(8)), max_new_tokens=8)
    scheduler.add(request=a)
    scheduler.add(request=b)
    scheduler.schedule()  # 2 blocks each: the pool is full
    for request in (a, b):
        request.computed, request.output = 8, [1]
    batch = scheduler.schedule()  # a needs a third block: b is preempted
    assert batch == [(a, 1)]
    assert b.state == State.WAITING and b.computed == 0 and scheduler.waiting[0] is b
    assert "b" not in scheduler.blocks.tables and scheduler.preemptions == 1


def test_static_batching_admits_nothing_while_a_batch_runs():
    scheduler = _scheduler(static_batching=True)
    scheduler.add(request=Request(request_id="a", prompt=[1, 2], max_new_tokens=4))
    scheduler.schedule()
    scheduler.add(request=Request(request_id="b", prompt=[1, 2], max_new_tokens=4))
    assert [r.request_id for r, _ in scheduler.schedule()] == ["a"]
