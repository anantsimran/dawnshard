from training.viz.roofline import KernelPoint, plot_roofline


def test_plot_writes_a_png_and_points_know_their_intensity(tmp_path):
    points = [
        KernelPoint(name="matmul", flops=1e12, bytes_moved=1e9, seconds=0.01),
        KernelPoint(name="norm", flops=1e9, bytes_moved=1e9, seconds=0.002),
    ]
    assert points[0].intensity == 1000
    assert points[1].achieved == 5e11
    plot_roofline(points=points, peak_flops=1e14, bandwidth=1e12, out_path=tmp_path / "r.png")
    assert (tmp_path / "r.png").stat().st_size > 0
