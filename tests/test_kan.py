import pytest
import torch

from kanrec.kan import BASES, KANLinear, ResidualBlock, bspline_basis, n_params


def test_bspline_partition_of_unity_inside_grid():
    layer = KANLinear(3, 2, basis="bspline", grid_size=8, spline_order=3, grid_range=(-1, 1))
    x = torch.linspace(-0.999, 0.999, 1001)
    b = bspline_basis(x, layer.grid, 3)
    assert b.shape == (1001, 11)
    assert torch.allclose(b.sum(-1), torch.ones(1001), atol=1e-5)


def test_bspline_is_cubic_not_step():
    layer = KANLinear(1, 1, basis="bspline", grid_size=8, spline_order=3, grid_range=(-1, 1))
    x = torch.linspace(-0.999, 0.999, 1001)
    b = bspline_basis(x, layer.grid, 3)
    assert b.max() < 0.7
    assert (b.abs().amax(0) > 0).all()


def test_bspline_matches_closed_form_uniform_cubic():
    grid = torch.arange(-3, 12, dtype=torch.float64)
    x = torch.tensor([4.25], dtype=torch.float64)
    b = bspline_basis(x, grid, 3)[0]
    u = 0.25
    expected = torch.tensor([(1 - u) ** 3 / 6, (3 * u**3 - 6 * u**2 + 4) / 6,
                             (-3 * u**3 + 3 * u**2 + 3 * u + 1) / 6, u**3 / 6], dtype=torch.float64)
    assert torch.allclose(b[4:8], expected)


@pytest.mark.parametrize("basis", BASES)
def test_shapes_and_gradients(basis):
    layer = KANLinear(16, 8, basis=basis)
    x = torch.randn(4, 5, 16, requires_grad=True)
    y = layer(x)
    assert y.shape == (4, 5, 8)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert all(p.grad is not None for p in layer.parameters())


@pytest.mark.parametrize("basis", ["bspline", "rbf", "cheby"])
def test_param_count(basis):
    layer = KANLinear(16, 8, basis=basis, grid_size=8, spline_order=3)
    n_basis = {"bspline": 11, "rbf": 8, "cheby": 9}[basis]
    assert n_params(layer) == 16 * 8 + 8 + 16 * 8 * n_basis


@pytest.mark.parametrize("basis", BASES)
def test_mlp_control_is_param_matched(basis):
    kan = ResidualBlock(64, kind="kan", basis=basis)
    mlp = ResidualBlock(64, kind="mlp", basis=basis)
    assert abs(n_params(kan) - n_params(mlp)) / n_params(kan) < 0.02


def test_edge_function_matches_forward():
    layer = KANLinear(4, 3, basis="rbf")
    xs = torch.linspace(-1.5, 1.5, 7)
    x = torch.zeros(7, 4)
    x[:, 2] = xs
    with torch.no_grad():
        zero_out = layer(torch.zeros(1, 4))[0, 1]
        full = layer(x)[:, 1]
        edge = layer.edge_function(1, 2, xs)
        edge0 = layer.edge_function(1, 2, torch.zeros(1))
    assert torch.allclose(full - zero_out, edge - edge0, atol=1e-5)


@pytest.mark.parametrize("kind", ["kan", "mlp"])
@pytest.mark.parametrize("basis", BASES)
def test_zero_init_block_is_identity(kind, basis):
    block = ResidualBlock(32, kind=kind, basis=basis)
    x = torch.randn(5, 32)
    assert torch.allclose(block(x), x)
