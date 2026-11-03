import torch

from src.models import GAT, GraphSAGE


def fake_graph():
    x = torch.randn(10, 41)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    return x, edge_index


def test_graphsage_output_shape():
    x, edge_index = fake_graph()
    assert GraphSAGE()(x, edge_index).shape == (10, 2)


def test_gat_output_shape():
    x, edge_index = fake_graph()
    assert GAT()(x, edge_index).shape == (10, 2)


def test_graphsage_loads_weights():
    model = GraphSAGE()
    model.load_state_dict(torch.load("models/GraphSAGE.pt", map_location="cpu"))


def test_gat_loads_weights():
    model = GAT()
    model.load_state_dict(torch.load("models/GAT.pt", map_location="cpu"))