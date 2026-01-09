import pytest
from unittest.mock import Mock, patch, MagicMock

from pyreft.utils import ReftType, TaskType, get_reft_model


class TestReftType:
    def test_loreft_value(self):
        assert ReftType.LOREFT == "LOREFT"
        assert ReftType.LOREFT.value == "LOREFT"
        
    def test_noreft_value(self):
        assert ReftType.NLOREFT == "NOREFT"
        assert ReftType.NLOREFT.value == "NOREFT"
        
    def test_is_string_enum(self):
        assert isinstance(ReftType.LOREFT, str)
        assert isinstance(ReftType.NLOREFT, str)


class TestTaskType:
    def test_seq_cls_value(self):
        assert TaskType.SEQ_CLS == "SEQ_CLS"
        assert TaskType.SEQ_CLS.value == "SEQ_CLS"
        
    def test_causal_lm_value(self):
        assert TaskType.CAUSAL_LM == "CAUSAL_LM"
        assert TaskType.CAUSAL_LM.value == "CAUSAL_LM"
        
    def test_is_string_enum(self):
        assert isinstance(TaskType.SEQ_CLS, str)
        assert isinstance(TaskType.CAUSAL_LM, str)


class TestGetReftModel:
    @patch("pyreft.utils.ReftModel")
    def test_get_reft_model_default_params(self, mock_reft_model_class):
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_reft_config = Mock()
        
        mock_reft_model = Mock()
        mock_reft_model_class.return_value = mock_reft_model
        
        result = get_reft_model(mock_model, mock_reft_config)
        
        mock_reft_model_class.assert_called_once_with(mock_reft_config, mock_model)
        mock_reft_model.set_device.assert_called_once_with(mock_model.device)
        mock_reft_model.disable_model_gradients.assert_called_once()
        assert result == mock_reft_model
        
    @patch("pyreft.utils.ReftModel")
    def test_get_reft_model_no_set_device(self, mock_reft_model_class):
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_reft_config = Mock()
        
        mock_reft_model = Mock()
        mock_reft_model_class.return_value = mock_reft_model
        
        result = get_reft_model(mock_model, mock_reft_config, set_device=False)
        
        mock_reft_model.set_device.assert_not_called()
        mock_reft_model.disable_model_gradients.assert_called_once()
        
    @patch("pyreft.utils.ReftModel")
    def test_get_reft_model_keep_model_grads(self, mock_reft_model_class):
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_reft_config = Mock()
        
        mock_reft_model = Mock()
        mock_reft_model_class.return_value = mock_reft_model
        
        result = get_reft_model(mock_model, mock_reft_config, disable_model_grads=False)
        
        mock_reft_model.set_device.assert_called_once()
        mock_reft_model.disable_model_gradients.assert_not_called()
        
    @patch("pyreft.utils.ReftModel")
    def test_get_reft_model_all_params_false(self, mock_reft_model_class):
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_reft_config = Mock()
        
        mock_reft_model = Mock()
        mock_reft_model_class.return_value = mock_reft_model
        
        result = get_reft_model(
            mock_model, 
            mock_reft_config, 
            set_device=False, 
            disable_model_grads=False
        )
        
        mock_reft_model.set_device.assert_not_called()
        mock_reft_model.disable_model_gradients.assert_not_called()
