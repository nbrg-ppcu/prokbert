# coding=utf-8
import warnings
import logging
from typing import Optional, Tuple, Union
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from transformers import MegatronBertConfig, MegatronBertModel, MegatronBertForMaskedLM, MegatronBertPreTrainedModel, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils.hub import cached_file
import math
#from prokbert.training_utils import compute_metrics_eval_prediction


def l2_norm(input, axis=1, epsilon=1e-12):
    norm = torch.norm(input, 2, axis, True)
    norm = torch.clamp(norm, min=epsilon)  # Avoid zero division
    output = torch.div(input, norm)
    return output

def initialize_linear_kaiming(layer: nn.Linear):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity='linear')
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

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

class ProkBertConfigCurr(ProkBertConfig):
    model_type = "prokbert"

    def __init__(
        self,
        bert_base_model = "neuralbioinfo/prokbert-mini",
        curricular_face_m = 0.5,
        curricular_face_s=64.,
        curricular_num_labels = 2,
        curriculum_hidden_size = -1, 
        classification_dropout_rate = 0.0, 
        **kwargs,
    ):
        super().__init__( **kwargs)
        self.curricular_num_labels = curricular_num_labels
        self.curricular_face_m = curricular_face_m
        self.curricular_face_s = curricular_face_s
        self.classification_dropout_rate = classification_dropout_rate
        self.bert_base_model = bert_base_model
        self.curriculum_hidden_size = curriculum_hidden_size

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


class CurricularFace(nn.Module):
    def __init__(self, in_features, out_features, m=0.5, s=64.):
        super(CurricularFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer('t', torch.zeros(1))
        initialize_linear_kaiming(self.kernel)

    def forward(self, embeddings, label):
        # Normalize embeddings and the classifier kernel
        embeddings = l2_norm(embeddings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        # Compute cosine similarity between embeddings and kernel columns
        cos_theta = torch.mm(embeddings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        
        # Clone original cosine values (used later for analysis if needed)
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        
        # Get the cosine values corresponding to the ground-truth classes
        target_logit = cos_theta[torch.arange(0, embeddings.size(0)), label].view(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target + margin)
        
        # Create a mask for positions where the cosine similarity exceeds the modified value
        mask = (cos_theta > cos_theta_m) #.to(dtype=torch.uint8)
        
        # Apply the margin condition: for values greater than threshold, use cosine with margin;
        # otherwise subtract a fixed term.
        final_target_logit = torch.where(target_logit > self.threshold, 
                                         cos_theta_m, 
                                         target_logit - self.mm)
        
        # Update the buffer 't' (used to control the weight of hard examples)
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        
        # For the positions in the mask, re-scale the logits
        try:
            hard_example = cos_theta[mask]
        except Exception as e:
            print("Label max")
            print(torch.max(label))
            print("Shapes:")
            print(embeddings.shape)
            print(label.shape)
            hard_example = cos_theta[mask]
            
        cos_theta[mask] = hard_example * (self.t + hard_example)
        
        # Replace the logits of the target classes with the modified target logit
        final_target_logit = final_target_logit.to(cos_theta.dtype)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output, origin_cos * self.s

class ProkBertForCurricularClassification(ProkBertPreTrainedModel):
    config_class = ProkBertConfigCurr
    base_model_prefix = "bert"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = ProkBertModel(config)
        
        # A weighting layer for pooling the sequence output
        self.weighting_layer = nn.Linear(self.config.hidden_size, 1)
        self.dropout = nn.Dropout(self.config.classification_dropout_rate)
        
        if config.curriculum_hidden_size != -1:
            self.linear = nn.Linear(self.config.hidden_size, config.curriculum_hidden_size)
            
            # Replace the simple classifier with the CurricularFace head.
            # Defaults m=0.5 and s=64 are used, but these can be adjusted if needed.
            self.curricular_face = CurricularFace(config.curriculum_hidden_size, 
                                                self.config.curricular_num_labels,
                                                m=self.config.curricular_face_m,
                                                s=self.config.curricular_face_s)
        else:
            self.linear = nn.Identity()
            self.curricular_face = CurricularFace(self.config.hidden_size, 
                                                self.config.curricular_num_labels,
                                                m=self.config.curricular_face_m,
                                                s=self.config.curricular_face_s)
        

        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.post_init()

    def _init_weights(self, module: nn.Module):
        # first let the base class init everything else
        super()._init_weights(module)

        # then catch our pooling head and zero it
        if module is getattr(self, "weighting_layer", None):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        if module is getattr(self, "linear", None):
            initialize_linear_kaiming(self.linear)
        
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get the outputs from the base ProkBert model
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
        sequence_output = outputs[0]  # (batch_size, seq_length, hidden_size)
        
        # Pool the sequence output using a learned weighting (attention-like)
        weights = self.weighting_layer(sequence_output)  # (batch_size, seq_length, 1)
        # Ensure mask shape matches
        if attention_mask.dim() == 2:
            mask = attention_mask
        elif attention_mask.dim() == 4:
            mask = attention_mask.squeeze(1).squeeze(1)  # (batch_size, seq_length)
        else:
            raise ValueError(f"Unexpected attention_mask shape {attention_mask.shape}")

        # Apply mask (masked positions -> -inf before softmax)
        weights = weights.masked_fill(mask == 0, float('-inf'))

        # Normalize
        weights = torch.nn.functional.softmax(weights, dim=1)  # (batch_size, seq_length)

        # Weighted pooling
        weights = weights.unsqueeze(-1)                        # (batch_size, seq_length, 1)        
        pooled_output = torch.sum(weights * sequence_output, dim=1)  # (batch_size, hidden_size)
        # Classifier head
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.linear(pooled_output)

        # CurricularFace requires the embeddings and the corresponding labels.
        # Note: During inference (labels is None), we just return l2 norm of bert part of the model
        if labels is None:
            return l2_norm(pooled_output, axis = 1) 
        else:
            logits, origin_cos = self.curricular_face(pooled_output, labels)
        
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels.view(-1))
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )