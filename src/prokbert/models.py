# coding=utf-8
import warnings
import logging

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MegatronBertConfig, MegatronBertModel



class BertForBinaryClassificationWithPooling(nn.Module):
    def __init__(self, base_model: MegatronBertModel):        
        super(BertForBinaryClassificationWithPooling, self).__init__()
        self.base_model = base_model
        self.base_model_config_dict = base_model.config.to_dict()
        self.hidden_size = self.base_model_config_dict['hidden_size']
        self.dropout_rate = self.base_model_config_dict['hidden_dropout_prob']

        self.weighting_layer = nn.Linear(self.hidden_size, 1)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, 2)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get sequence representations from base model
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        
        # Compute weights for each position in the sequence
        weights = self.weighting_layer(sequence_output)
        weights = torch.nn.functional.softmax(weights, dim=1)
        
        # Compute weighted sum
        pooled_output = torch.sum(weights * sequence_output, dim=1)
        
        # Classification head
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Prepare the output as a dictionary
        output = {"logits": logits}
        
        # If labels are provided, compute the loss. This is useful during training.
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            output["loss"] = loss        
        return output

    def save_pretrained(self, save_directory):
        """
        Save the model weights and configuration in a directory.

        Args:
            save_directory (str): Directory where the model and configuration can be saved.
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        self.base_model.config.save_pretrained(save_directory)


    @classmethod
    def from_pretrained(cls, pretrained_directory, config=None):
        """
        Load the model weights and configuration from a directory.

        Args:
            pretrained_directory (str): Directory where the model and configuration were saved.

        Returns:
            model: Instance of BertForBinaryClassificationWithPooling.
        """
        # Load the base model's configuration
        if config is None:
            megatron_bert_config = MegatronBertConfig.from_pretrained(pretrained_directory)
            base_model = MegatronBertModel(config=megatron_bert_config)
        else:
            base_model = MegatronBertModel(config=config)

      # Create model instance using loaded configuration
        model = cls(base_model=base_model)
        model_path = os.path.join(pretrained_directory, "pytorch_model.bin")
        model.load_state_dict(torch.load(model_path))
        
        return model

    
