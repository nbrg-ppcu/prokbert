# coding=utf-8
import warnings
import logging

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MegatronBertConfig, MegatronBertModel
from transformers.utils.hub import cached_file

import transformers

class BertForBinaryClassificationWithPooling(nn.Module):
    """
    ProkBERT model for binary classification with custom pooling.

    This model extends a pre-trained `MegatronBertModel` by adding a weighting layer
    to compute a weighted sum over the sequence outputs, followed by a classifier.

    Attributes:
        base_model (MegatronBertModel): The base BERT model.
        weighting_layer (nn.Linear): Linear layer to compute weights for each token.
        dropout (nn.Dropout): Dropout layer.
        classifier (nn.Linear): Linear layer for classification.
    """    
    def __init__(self, base_model: MegatronBertModel):
        """
        Initialize the BertForBinaryClassificationWithPooling model.

        Args:
            base_model (MegatronBertModel): A pre-trained `MegatronBertModel` instance.
        """
                        
        super(BertForBinaryClassificationWithPooling, self).__init__()
        self.base_model = base_model
        self.base_model_config_dict = base_model.config.to_dict()
        self.hidden_size = self.base_model_config_dict['hidden_size']
        self.dropout_rate = self.base_model_config_dict['hidden_dropout_prob']

        self.weighting_layer = nn.Linear(self.hidden_size, 1)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, 2)

    def forward(self, input_ids, attention_mask=None, labels=None, output_hidden_states=False, output_pooled_output=False):
        # Modified call to base model to include output_hidden_states
        outputs = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=output_hidden_states)
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
        
        # Include hidden states in output if requested
        if output_hidden_states:
            output["hidden_states"] = outputs.hidden_states
        if output_pooled_output:
            output["pooled_output"] = pooled_output
        
        # If labels are provided, compute the loss
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
        print('The save pretrained is called!')
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        print(f'The save directory is: {save_directory}')        
        self.base_model.config.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load the model weights and configuration from a local directory or Hugging Face Hub.

        Args:
            pretrained_model_name_or_path (str): Directory path where the model and configuration were saved, or name of the model in Hugging Face Hub.

        Returns:
            model: Instance of BertForBinaryClassificationWithPooling.
        """
        # Determine if the path is local or from Hugging Face Hub
        if os.path.exists(pretrained_model_name_or_path):
            # Path is local
            if 'config' in kwargs:
                print('Config is in the parameters')
                config = kwargs['config']
                  
            else:                
                config = MegatronBertConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
            base_model = MegatronBertModel(config=config)
            model = cls(base_model=base_model)
            model_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        else:
            # Path is from Hugging Face Hub
            config = kwargs.pop('config', None)
            if config is None:
                config = MegatronBertConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

            base_model = MegatronBertModel(config=config)
            model = cls(base_model=base_model)
            model_file = cached_file(pretrained_model_name_or_path, "pytorch_model.bin")
            model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu'), weights_only=True))

        return model
