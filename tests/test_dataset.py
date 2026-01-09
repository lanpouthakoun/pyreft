import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from collections import defaultdict

from pyreft.dataset import (
    parse_positions,
    get_intervention_locations,
    ReftDataCollator,
    IGNORE_INDEX,
)


class TestParsePositions:
    def test_parse_first_n_only(self):
        first_n, last_n = parse_positions("f3")
        assert first_n == 3
        assert last_n == 0
        
    def test_parse_last_n_only(self):
        first_n, last_n = parse_positions("l5")
        assert first_n == 0
        assert last_n == 5
        
    def test_parse_first_and_last(self):
        first_n, last_n = parse_positions("f3+l5")
        assert first_n == 3
        assert last_n == 5
        
    def test_parse_equal_first_and_last(self):
        first_n, last_n = parse_positions("f7+l7")
        assert first_n == 7
        assert last_n == 7
        
    def test_parse_single_position(self):
        first_n, last_n = parse_positions("f1")
        assert first_n == 1
        assert last_n == 0
        
    def test_parse_large_numbers(self):
        first_n, last_n = parse_positions("f100+l200")
        assert first_n == 100
        assert last_n == 200


class TestGetInterventionLocations:
    def test_basic_first_n_positions(self):
        locations = get_intervention_locations(
            last_position=20,
            first_n=3,
            last_n=0,
            num_interventions=2,
            share_weights=True
        )
        
        assert len(locations) == 2
        assert locations[0][:3] == [0, 1, 2]
        
    def test_basic_last_n_positions(self):
        locations = get_intervention_locations(
            last_position=20,
            first_n=0,
            last_n=3,
            num_interventions=2,
            share_weights=True
        )
        
        assert len(locations) == 2
        assert 17 in locations[0]
        assert 18 in locations[0]
        assert 19 in locations[0]
        
    def test_first_and_last_positions_shared_weights(self):
        locations = get_intervention_locations(
            last_position=20,
            first_n=2,
            last_n=2,
            num_interventions=2,
            share_weights=True
        )
        
        assert len(locations) == 2
        assert 0 in locations[0]
        assert 1 in locations[0]
        assert 18 in locations[0]
        assert 19 in locations[0]
        
    def test_first_and_last_positions_separate_weights(self):
        locations = get_intervention_locations(
            last_position=20,
            first_n=2,
            last_n=2,
            num_interventions=4,
            share_weights=False
        )
        
        assert len(locations) == 4
        assert locations[0] == locations[1]
        assert locations[2] == locations[3]
        
    def test_padding_when_sequence_too_short(self):
        locations = get_intervention_locations(
            last_position=4,
            first_n=5,
            last_n=5,
            num_interventions=2,
            share_weights=True,
            pad_mode="first"
        )
        
        assert len(locations) == 2
        assert -1 in locations[0]
        
    def test_padding_mode_last(self):
        locations = get_intervention_locations(
            last_position=4,
            first_n=5,
            last_n=0,
            num_interventions=2,
            share_weights=True,
            pad_mode="last"
        )
        
        assert len(locations) == 2
        assert 4 in locations[0]
        
    def test_with_positions_string(self):
        locations = get_intervention_locations(
            last_position=20,
            positions="f3+l3",
            num_interventions=2,
            share_weights=True
        )
        
        assert len(locations) == 2
        
    def test_num_interventions(self):
        locations = get_intervention_locations(
            last_position=20,
            first_n=2,
            last_n=2,
            num_interventions=6,
            share_weights=True
        )
        
        assert len(locations) == 6
        for loc in locations:
            assert loc == locations[0]


class TestReftDataCollator:
    def test_call_truncates_intervention_locations(self):
        mock_data_collator = Mock()
        mock_data_collator.return_value = {
            "input_ids": torch.zeros(2, 10),
            "attention_mask": torch.ones(2, 10),
            "intervention_locations": torch.zeros(2, 4, 20),
        }
        
        collator = ReftDataCollator(data_collator=mock_data_collator)
        
        instances = [{"input_ids": torch.zeros(10)}]
        result = collator(instances)
        
        assert result["intervention_locations"].shape[-1] == 10
        
    def test_call_preserves_other_fields(self):
        mock_data_collator = Mock()
        mock_data_collator.return_value = {
            "input_ids": torch.zeros(2, 10),
            "attention_mask": torch.ones(2, 10),
            "labels": torch.zeros(2, 10),
            "intervention_locations": torch.zeros(2, 4, 10),
        }
        
        collator = ReftDataCollator(data_collator=mock_data_collator)
        
        instances = [{"input_ids": torch.zeros(10)}]
        result = collator(instances)
        
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result
        assert result["input_ids"].shape == (2, 10)


class TestIgnoreIndex:
    def test_ignore_index_value(self):
        assert IGNORE_INDEX == -100
