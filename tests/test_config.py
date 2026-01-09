import pytest
from unittest.mock import patch, MagicMock

from pyreft.config import ReftConfig


class TestReftConfig:
    @patch("pyreft.config.pv.IntervenableConfig.__init__")
    def test_initialization(self, mock_parent_init):
        mock_parent_init.return_value = None
        
        config = ReftConfig(
            representations=[{
                "layer": 0,
                "component": "block_output",
                "low_rank_dimension": 8,
            }]
        )
        
        mock_parent_init.assert_called_once()
        
    @patch("pyreft.config.pv.IntervenableConfig.__init__")
    def test_initialization_with_multiple_representations(self, mock_parent_init):
        mock_parent_init.return_value = None
        
        representations = [
            {"layer": 0, "component": "block_output", "low_rank_dimension": 8},
            {"layer": 1, "component": "block_output", "low_rank_dimension": 8},
        ]
        
        config = ReftConfig(representations=representations)
        
        mock_parent_init.assert_called_once()
        
    @patch("pyreft.config.pv.IntervenableConfig.__init__")
    def test_initialization_empty_kwargs(self, mock_parent_init):
        mock_parent_init.return_value = None
        
        config = ReftConfig()
        
        mock_parent_init.assert_called_once_with()
