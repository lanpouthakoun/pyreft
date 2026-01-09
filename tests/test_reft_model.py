import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock

from pyreft.reft_model import count_parameters, ReftModel


class TestCountParameters:
    def test_count_parameters_simple_model(self):
        model = nn.Linear(10, 5)
        count = count_parameters(model)
        
        expected = 10 * 5 + 5
        assert count == expected
        
    def test_count_parameters_no_grad(self):
        model = nn.Linear(10, 5)
        for param in model.parameters():
            param.requires_grad = False
            
        count = count_parameters(model)
        assert count == 0
        
    def test_count_parameters_mixed_grad(self):
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Linear(5, 2)
        )
        model[0].weight.requires_grad = False
        model[0].bias.requires_grad = False
        
        count = count_parameters(model)
        
        expected = 5 * 2 + 2
        assert count == expected
        
    def test_count_parameters_empty_model(self):
        model = nn.Sequential()
        count = count_parameters(model)
        assert count == 0


class TestReftModel:
    @patch("pyreft.reft_model.pv.IntervenableModel.__init__")
    def test_initialization(self, mock_parent_init):
        mock_parent_init.return_value = None
        
        mock_config = Mock()
        mock_model = Mock()
        
        reft_model = ReftModel(mock_config, mock_model)
        
        mock_parent_init.assert_called_once_with(mock_config, mock_model)
        
    @patch("pyreft.reft_model.pv.IntervenableModel.load")
    def test_load_static_method(self, mock_parent_load):
        mock_intervenable = Mock()
        mock_intervenable.config = Mock()
        mock_intervenable.model = Mock()
        mock_parent_load.return_value = mock_intervenable
        
        with patch.object(ReftModel, "_convert_to_reft_model") as mock_convert:
            mock_reft_model = Mock()
            mock_convert.return_value = mock_reft_model
            
            result = ReftModel.load("path/to/model")
            
            mock_parent_load.assert_called_once_with("path/to/model")
            mock_convert.assert_called_once_with(mock_intervenable)
            assert result == mock_reft_model
            
    def test_convert_to_reft_model(self):
        mock_intervenable = Mock()
        mock_intervenable.config = Mock()
        mock_intervenable.model = Mock()
        
        with patch("pyreft.reft_model.pv.IntervenableModel.__init__", return_value=None):
            result = ReftModel._convert_to_reft_model(mock_intervenable)
            
            assert isinstance(result, ReftModel)
            
    @patch("pyreft.reft_model.pv.IntervenableModel.__init__")
    def test_print_trainable_parameters(self, mock_parent_init):
        mock_parent_init.return_value = None
        
        mock_config = Mock()
        mock_model = Mock()
        
        mock_param1 = Mock()
        mock_param1.numel.return_value = 100
        mock_param1.requires_grad = True
        
        mock_param2 = Mock()
        mock_param2.numel.return_value = 200
        mock_param2.requires_grad = False
        
        mock_model.parameters.return_value = [mock_param1, mock_param2]
        
        reft_model = ReftModel(mock_config, mock_model)
        reft_model.interventions = {}
        reft_model._intervention_reverse_link = {}
        reft_model.model = mock_model
        
        reft_model.print_trainable_parameters()
        
    @patch("pyreft.reft_model.pv.IntervenableModel.__init__")
    def test_print_trainable_parameters_with_interventions(self, mock_parent_init):
        mock_parent_init.return_value = None
        
        mock_config = Mock()
        mock_model = Mock()
        
        mock_param = Mock()
        mock_param.numel.return_value = 100
        mock_param.requires_grad = True
        mock_model.parameters.return_value = [mock_param]
        
        reft_model = ReftModel(mock_config, mock_model)
        reft_model.model = mock_model
        reft_model._intervention_reverse_link = {}
        
        mock_intervention = Mock()
        mock_intervention_param = Mock()
        mock_intervention_param.numel.return_value = 50
        mock_intervention_param.requires_grad = True
        mock_intervention.parameters.return_value = [mock_intervention_param]
        
        with patch("pyreft.reft_model.pv.TrainableIntervention") as mock_trainable:
            reft_model.interventions = {"key1": mock_intervention}
            
            reft_model.print_trainable_parameters()
