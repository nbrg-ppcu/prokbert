# coding=utf-8
import warnings
import logging
from typing import Optional, Tuple, Union
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MegatronBertConfig, MegatronBertModel, MegatronBertForMaskedLM, MegatronBertPreTrainedModel, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils.hub import cached_file
#from prokbert.training_utils import compute_metrics_eval_prediction

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




class ProkBertConfig(MegatronBertConfig):
    model_type = "prokbert"

    def __init__(
        self,
        kmer: int = 6,
        shift: int = 1,
        num_class_labels: int = 2,
        classification_dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kmer = kmer
        self.shift = shift
        self.num_class_labels = num_class_labels
        self.classification_dropout_rate = classification_dropout_rate




class ProkBertClassificationConfig(ProkBertConfig):
    model_type = "prokbert"
    def __init__(
        self,
        num_labels: int = 2,
        classification_dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Ide jön majd némi extra lépés, egyelőre csak próbálkozunk a sima configgal. 
        self.num_labels = num_labels
        self.classification_dropout_rate = classification_dropout_rate

class ProkBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ProkBertConfig
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()




class ProkBertModel(MegatronBertModel):
    config_class = ProkBertConfig

    def __init__(self, config: ProkBertConfig, **kwargs):
        if not isinstance(config, ProkBertConfig):
            raise ValueError(f"Expected `ProkBertConfig`, got {config.__class__.__module__}.{config.__class__.__name__}")

        super().__init__(config, **kwargs)
        self.config = config
        # One should check if it is a prper prokbert config, if not crafting one.


class ProkBertForMaskedLM(MegatronBertForMaskedLM):
    config_class = ProkBertConfig

    def __init__(self, config: ProkBertConfig, **kwargs):
        if not isinstance(config, ProkBertConfig):
            raise ValueError(f"Expected `ProkBertConfig`, got {config.__class__.__module__}.{config.__class__.__name__}")

        super().__init__(config, **kwargs)
        self.config = config
        # One should check if it is a prper prokbert config, if not crafting one.


class ProkBertForSequenceClassification(ProkBertPreTrainedModel):
    config_class = ProkBertConfig
    base_model_prefix = "bert"

    def __init__(self, config):

        super().__init__(config)
        self.config = config
        self.bert = ProkBertModel(config)                
        self.weighting_layer = nn.Linear(self.config.hidden_size, 1)
        self.dropout = nn.Dropout(self.config.classification_dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_class_labels)
        self.loss_fct = torch.nn.CrossEntropyLoss()
        
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple, SequenceClassifierOutput]:
            r"""
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_class_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_class_labels > 1` a classification loss is computed (Cross-Entropy).
            """
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = outputs[0]
            
            # Compute weights for each position in the sequence
            weights = self.weighting_layer(sequence_output)
            weights = torch.nn.functional.softmax(weights, dim=1)            
            # Compute weighted sum
            pooled_output = torch.sum(weights * sequence_output, dim=1)            
            # Classification head
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            loss = None
            if labels is not None:
                loss = self.loss_fct(logits.view(-1, self.config.num_class_labels), labels.view(-1))

            classification_output = SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
            return classification_output

