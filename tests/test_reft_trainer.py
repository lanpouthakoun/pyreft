import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from torch.utils.data import DataLoader

from pyreft.reft_trainer import (
    ReftDataCollator,
    make_data_collator,
    make_dataloader,
    ReftTrainer,
    ReftTrainerForCausalLM,
    ReftTrainerForSequenceClassification,
)


class TestReftDataCollatorTrainer:
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
        
    def test_call_preserves_input_ids_shape(self):
        mock_data_collator = Mock()
        mock_data_collator.return_value = {
            "input_ids": torch.zeros(4, 15),
            "attention_mask": torch.ones(4, 15),
            "intervention_locations": torch.zeros(4, 2, 15),
        }
        
        collator = ReftDataCollator(data_collator=mock_data_collator)
        
        instances = [{"input_ids": torch.zeros(15)}]
        result = collator(instances)
        
        assert result["input_ids"].shape == (4, 15)


class TestMakeDataCollator:
    @patch("pyreft.reft_trainer.DataCollatorForSeq2Seq")
    def test_make_data_collator(self, mock_collator_class):
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_collator_instance = Mock()
        mock_collator_class.return_value = mock_collator_instance
        
        result = make_data_collator(mock_tokenizer, mock_model)
        
        mock_collator_class.assert_called_once_with(
            tokenizer=mock_tokenizer,
            model=mock_model,
            label_pad_token_id=-100,
            padding="longest",
            max_length=2048,
        )
        assert isinstance(result, ReftDataCollator)
        assert result.data_collator == mock_collator_instance


class TestMakeDataloader:
    def test_make_dataloader_with_shuffle(self):
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)
        mock_collate_fn = Mock()
        
        dataloader = make_dataloader(
            dataset=mock_dataset,
            batch_size=2,
            collate_fn=mock_collate_fn,
            shuffle=True
        )
        
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 2
        
    def test_make_dataloader_without_shuffle(self):
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)
        mock_collate_fn = Mock()
        
        dataloader = make_dataloader(
            dataset=mock_dataset,
            batch_size=4,
            collate_fn=mock_collate_fn,
            shuffle=False
        )
        
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 4
        
    def test_make_dataloader_with_sampler(self):
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)
        mock_collate_fn = Mock()
        mock_sampler = Mock()
        
        dataloader = make_dataloader(
            dataset=mock_dataset,
            batch_size=2,
            collate_fn=mock_collate_fn,
            shuffle=False,
            sampler=mock_sampler
        )
        
        assert isinstance(dataloader, DataLoader)


class TestReftTrainer:
    @patch("pyreft.reft_trainer.Trainer.__init__")
    def test_save_model_creates_directory(self, mock_trainer_init):
        mock_trainer_init.return_value = None
        
        trainer = ReftTrainer.__new__(ReftTrainer)
        trainer.model = Mock()
        
        with patch("os.path.exists", return_value=False), \
             patch("os.makedirs") as mock_makedirs, \
             patch("os.listdir", return_value=[]), \
             patch("pyreft.reft_trainer.dist.is_initialized", return_value=False):
            
            trainer.save_model("/tmp/test_output")
            
            mock_makedirs.assert_called_once_with("/tmp/test_output")
            trainer.model.save_intervention.assert_called_once()
            
    @patch("pyreft.reft_trainer.Trainer.__init__")
    def test_save_model_skips_existing_directory(self, mock_trainer_init):
        mock_trainer_init.return_value = None
        
        trainer = ReftTrainer.__new__(ReftTrainer)
        trainer.model = Mock()
        
        with patch("os.path.exists", return_value=True), \
             patch("os.listdir", return_value=["file.bin"]), \
             patch("pyreft.reft_trainer.dist.is_initialized", return_value=False):
            
            trainer.save_model("/tmp/test_output")
            
            trainer.model.save_intervention.assert_not_called()
            
    @patch("pyreft.reft_trainer.Trainer.__init__")
    def test_load_best_model(self, mock_trainer_init):
        mock_trainer_init.return_value = None
        
        trainer = ReftTrainer.__new__(ReftTrainer)
        trainer.model = Mock()
        trainer.state = Mock()
        trainer.state.best_model_checkpoint = "/tmp/best_checkpoint"
        trainer.state.best_metric = 0.95
        
        trainer._load_best_model()
        
        trainer.model.load_intervention.assert_called_once_with(
            "/tmp/best_checkpoint/intervenable_model",
            include_model=True
        )
        
    @patch("pyreft.reft_trainer.Trainer.__init__")
    def test_load_from_checkpoint(self, mock_trainer_init):
        mock_trainer_init.return_value = None
        
        trainer = ReftTrainer.__new__(ReftTrainer)
        trainer.model = Mock()
        
        trainer._load_from_checkpoint("/tmp/checkpoint")
        
        trainer.model.load_intervention.assert_called_once_with(
            "/tmp/checkpoint/intervenable_model",
            include_model=True
        )
        
    @patch("pyreft.reft_trainer.Trainer.__init__")
    def test_compute_loss_with_intervention_locations(self, mock_trainer_init):
        mock_trainer_init.return_value = None
        
        trainer = ReftTrainer.__new__(ReftTrainer)
        
        mock_intervenable = Mock()
        mock_output = Mock()
        mock_output.loss = torch.tensor(0.5)
        mock_intervenable.return_value = (None, mock_output)
        
        inputs = {
            "input_ids": torch.zeros(2, 10),
            "attention_mask": torch.ones(2, 10),
            "labels": torch.zeros(2, 10),
            "intervention_locations": torch.zeros(2, 4, 10),
        }
        
        loss = trainer.compute_loss(mock_intervenable, inputs)
        
        assert loss == mock_output.loss
        
    @patch("pyreft.reft_trainer.Trainer.__init__")
    def test_compute_loss_returns_outputs(self, mock_trainer_init):
        mock_trainer_init.return_value = None
        
        trainer = ReftTrainer.__new__(ReftTrainer)
        
        mock_intervenable = Mock()
        mock_output = Mock()
        mock_output.loss = torch.tensor(0.5)
        mock_intervenable.return_value = (None, mock_output)
        
        inputs = {
            "input_ids": torch.zeros(2, 10),
            "attention_mask": torch.ones(2, 10),
            "labels": torch.zeros(2, 10),
            "intervention_locations": torch.zeros(2, 4, 10),
        }
        
        result = trainer.compute_loss(mock_intervenable, inputs, return_outputs=True)
        
        assert result == (mock_output, mock_output)


class TestReftTrainerForCausalLM:
    @patch("pyreft.reft_trainer.Trainer.__init__")
    def test_get_train_dataloader(self, mock_trainer_init):
        mock_trainer_init.return_value = None
        
        trainer = ReftTrainerForCausalLM.__new__(ReftTrainerForCausalLM)
        trainer.train_dataset = Mock()
        trainer.train_dataset.__len__ = Mock(return_value=10)
        trainer._train_batch_size = 2
        trainer.data_collator = Mock()
        
        dataloader = trainer.get_train_dataloader()
        
        assert isinstance(dataloader, DataLoader)


class TestReftTrainerForSequenceClassification:
    @patch("pyreft.reft_trainer.Trainer.__init__")
    def test_compute_loss_single_label(self, mock_trainer_init):
        mock_trainer_init.return_value = None
        
        trainer = ReftTrainerForSequenceClassification.__new__(ReftTrainerForSequenceClassification)
        
        mock_intervenable = Mock()
        mock_output = Mock()
        mock_output.logits = torch.randn(2, 3)
        mock_intervenable.return_value = (None, mock_output)
        
        trainer.model = Mock()
        trainer.model.model = Mock()
        trainer.model.model.config = Mock()
        trainer.model.model.config.problem_type = "single_label_classification"
        trainer.model.model.num_labels = 3
        
        inputs = {
            "input_ids": torch.zeros(2, 10),
            "attention_mask": torch.ones(2, 10),
            "labels": torch.tensor([0, 1]),
            "intervention_locations": torch.zeros(2, 4, 10),
        }
        
        loss = trainer.compute_loss(mock_intervenable, inputs)
        
        assert isinstance(loss, torch.Tensor)
        
    @patch("pyreft.reft_trainer.Trainer.__init__")
    def test_compute_loss_regression(self, mock_trainer_init):
        mock_trainer_init.return_value = None
        
        trainer = ReftTrainerForSequenceClassification.__new__(ReftTrainerForSequenceClassification)
        
        mock_intervenable = Mock()
        mock_output = Mock()
        mock_output.logits = torch.randn(2, 1)
        mock_intervenable.return_value = (None, mock_output)
        
        trainer.model = Mock()
        trainer.model.model = Mock()
        trainer.model.model.config = Mock()
        trainer.model.model.config.problem_type = "regression"
        trainer.model.model.num_labels = 1
        
        inputs = {
            "input_ids": torch.zeros(2, 10),
            "attention_mask": torch.ones(2, 10),
            "labels": torch.tensor([0.5, 0.8]),
            "intervention_locations": torch.zeros(2, 4, 10),
        }
        
        loss = trainer.compute_loss(mock_intervenable, inputs)
        
        assert isinstance(loss, torch.Tensor)
