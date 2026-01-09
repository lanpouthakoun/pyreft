import pytest
import torch
import torch.nn as nn

from pyreft.interventions import (
    LowRankRotateLayer,
    LoreftIntervention,
    NoreftIntervention,
    ConsreftIntervention,
    LobireftIntervention,
    DireftIntervention,
    NodireftIntervention,
)


class TestLowRankRotateLayer:
    def test_initialization_with_orthogonal(self):
        n, m = 64, 8
        layer = LowRankRotateLayer(n, m, init_orth=True)
        
        assert layer.weight.shape == (n, m)
        assert layer.weight.requires_grad
        
    def test_initialization_without_orthogonal(self):
        n, m = 64, 8
        layer = LowRankRotateLayer(n, m, init_orth=False)
        
        assert layer.weight.shape == (n, m)
        assert layer.weight.requires_grad
        
    def test_forward_pass(self):
        n, m = 64, 8
        batch_size = 4
        seq_len = 10
        
        layer = LowRankRotateLayer(n, m, init_orth=True)
        x = torch.randn(batch_size, seq_len, n)
        
        output = layer(x)
        
        assert output.shape == (batch_size, seq_len, m)
        
    def test_forward_dtype_conversion(self):
        n, m = 64, 8
        layer = LowRankRotateLayer(n, m, init_orth=True)
        layer.weight.data = layer.weight.data.to(torch.float32)
        
        x = torch.randn(4, 10, n, dtype=torch.float64)
        output = layer(x)
        
        assert output.dtype == layer.weight.dtype


class TestLoreftIntervention:
    @pytest.fixture
    def intervention_kwargs(self):
        return {
            "embed_dim": 64,
            "low_rank_dimension": 8,
            "dtype": torch.float32,
            "dropout": 0.0,
        }
    
    def test_initialization(self, intervention_kwargs):
        intervention = LoreftIntervention(**intervention_kwargs)
        
        assert hasattr(intervention, "rotate_layer")
        assert hasattr(intervention, "learned_source")
        assert hasattr(intervention, "dropout")
        assert hasattr(intervention, "act_fn")
        
    def test_forward_pass(self, intervention_kwargs):
        intervention = LoreftIntervention(**intervention_kwargs)
        batch_size = 4
        seq_len = 10
        embed_dim = intervention_kwargs["embed_dim"]
        
        base = torch.randn(batch_size, seq_len, embed_dim)
        output = intervention(base)
        
        assert output.shape == base.shape
        assert output.dtype == base.dtype
        
    def test_forward_with_dropout(self, intervention_kwargs):
        intervention_kwargs["dropout"] = 0.5
        intervention = LoreftIntervention(**intervention_kwargs)
        
        base = torch.randn(4, 10, intervention_kwargs["embed_dim"])
        
        intervention.train()
        output_train = intervention(base)
        
        intervention.eval()
        output_eval = intervention(base)
        
        assert output_train.shape == base.shape
        assert output_eval.shape == base.shape
        
    def test_state_dict(self, intervention_kwargs):
        intervention = LoreftIntervention(**intervention_kwargs)
        state_dict = intervention.state_dict()
        
        assert "weight" in state_dict
        assert "bias" in state_dict
        assert "rotate_layer" in state_dict
        
    def test_load_state_dict(self, intervention_kwargs):
        intervention1 = LoreftIntervention(**intervention_kwargs)
        intervention2 = LoreftIntervention(**intervention_kwargs)
        
        state_dict = intervention1.state_dict()
        intervention2.load_state_dict(state_dict)
        
        assert torch.allclose(
            intervention1.rotate_layer.weight, 
            intervention2.rotate_layer.weight
        )


class TestNoreftIntervention:
    @pytest.fixture
    def intervention_kwargs(self):
        return {
            "embed_dim": 64,
            "low_rank_dimension": 8,
            "add_bias": True,
            "dtype": torch.float32,
            "dropout": 0.0,
        }
    
    def test_initialization(self, intervention_kwargs):
        intervention = NoreftIntervention(**intervention_kwargs)
        
        assert hasattr(intervention, "proj_layer")
        assert hasattr(intervention, "learned_source")
        assert hasattr(intervention, "dropout")
        assert hasattr(intervention, "act_fn")
        
    def test_forward_pass(self, intervention_kwargs):
        intervention = NoreftIntervention(**intervention_kwargs)
        batch_size = 4
        seq_len = 10
        embed_dim = intervention_kwargs["embed_dim"]
        
        base = torch.randn(batch_size, seq_len, embed_dim)
        output = intervention(base)
        
        assert output.shape == base.shape
        assert output.dtype == base.dtype
        
    def test_forward_without_bias(self, intervention_kwargs):
        intervention_kwargs["add_bias"] = False
        intervention = NoreftIntervention(**intervention_kwargs)
        
        base = torch.randn(4, 10, intervention_kwargs["embed_dim"])
        output = intervention(base)
        
        assert output.shape == base.shape


class TestConsreftIntervention:
    @pytest.fixture
    def intervention_kwargs(self):
        return {
            "embed_dim": 64,
            "low_rank_dimension": 8,
        }
    
    def test_initialization(self, intervention_kwargs):
        intervention = ConsreftIntervention(**intervention_kwargs)
        
        assert hasattr(intervention, "rotate_layer")
        assert hasattr(intervention, "learned_source")
        assert intervention.learned_source.shape == (intervention_kwargs["low_rank_dimension"],)
        
    def test_forward_pass(self, intervention_kwargs):
        intervention = ConsreftIntervention(**intervention_kwargs)
        batch_size = 4
        seq_len = 10
        embed_dim = intervention_kwargs["embed_dim"]
        
        base = torch.randn(batch_size, seq_len, embed_dim)
        output = intervention(base)
        
        assert output.shape == base.shape
        assert output.dtype == base.dtype


class TestLobireftIntervention:
    @pytest.fixture
    def intervention_kwargs(self):
        return {
            "embed_dim": 64,
            "low_rank_dimension": 8,
            "dropout": 0.0,
        }
    
    def test_initialization(self, intervention_kwargs):
        intervention = LobireftIntervention(**intervention_kwargs)
        
        assert hasattr(intervention, "rotate_layer")
        assert hasattr(intervention, "learned_source")
        assert hasattr(intervention, "dropout")
        
    def test_forward_pass(self, intervention_kwargs):
        intervention = LobireftIntervention(**intervention_kwargs)
        batch_size = 4
        seq_len = 10
        embed_dim = intervention_kwargs["embed_dim"]
        
        base = torch.randn(batch_size, seq_len, embed_dim)
        output = intervention(base)
        
        assert output.shape == base.shape
        assert output.dtype == base.dtype


class TestDireftIntervention:
    @pytest.fixture
    def intervention_kwargs(self):
        return {
            "embed_dim": 64,
            "low_rank_dimension": 8,
            "dtype": torch.float32,
            "dropout": 0.0,
        }
    
    def test_initialization(self, intervention_kwargs):
        intervention = DireftIntervention(**intervention_kwargs)
        
        assert hasattr(intervention, "rotate_layer")
        assert hasattr(intervention, "learned_source")
        assert hasattr(intervention, "dropout")
        assert hasattr(intervention, "act_fn")
        
    def test_forward_pass(self, intervention_kwargs):
        intervention = DireftIntervention(**intervention_kwargs)
        batch_size = 4
        seq_len = 10
        embed_dim = intervention_kwargs["embed_dim"]
        
        base = torch.randn(batch_size, seq_len, embed_dim)
        output = intervention(base)
        
        assert output.shape == base.shape
        assert output.dtype == base.dtype


class TestNodireftIntervention:
    @pytest.fixture
    def intervention_kwargs(self):
        return {
            "embed_dim": 64,
            "low_rank_dimension": 8,
            "add_bias": True,
            "dtype": torch.float32,
            "dropout": 0.0,
        }
    
    def test_initialization(self, intervention_kwargs):
        intervention = NodireftIntervention(**intervention_kwargs)
        
        assert hasattr(intervention, "proj_layer")
        assert hasattr(intervention, "learned_source")
        assert hasattr(intervention, "dropout")
        assert hasattr(intervention, "act_fn")
        
    def test_forward_pass(self, intervention_kwargs):
        intervention = NodireftIntervention(**intervention_kwargs)
        batch_size = 4
        seq_len = 10
        embed_dim = intervention_kwargs["embed_dim"]
        
        base = torch.randn(batch_size, seq_len, embed_dim)
        output = intervention(base)
        
        assert output.shape == base.shape
        assert output.dtype == base.dtype
        
    def test_forward_without_bias(self, intervention_kwargs):
        intervention_kwargs["add_bias"] = False
        intervention = NodireftIntervention(**intervention_kwargs)
        
        base = torch.randn(4, 10, intervention_kwargs["embed_dim"])
        output = intervention(base)
        
        assert output.shape == base.shape
