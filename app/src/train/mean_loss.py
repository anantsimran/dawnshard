def accumulate_loss(acc: dict, loss: float, batch_size: int) -> None:
    acc["accumulated_loss"] += loss * batch_size
    acc["count"] += batch_size


def compute_reduced_loss(acc: dict) -> float:
    return acc["accumulated_loss"] / acc["count"] if acc["count"] else 0.0
