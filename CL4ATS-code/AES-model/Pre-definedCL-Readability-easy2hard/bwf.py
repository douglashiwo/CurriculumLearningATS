
import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_outputs import SequenceClassifierOutput


_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-base-cased-finetuned-mrpc",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "TurkuNLP/bert-base-finnish-cased-v1",
    "TurkuNLP/bert-base-finnish-uncased-v1",
    "wietsedv/bert-base-dutch-cased",
    # See all BERT models at https://huggingface.co/models?filter=bert
]



'''

from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)


from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

from transformers.utils import logging

from transformers.models.bert.configuration_bert import BertConfig

'''




class BertWithFeatures(nn.Module):
    def __init__(self, BertCLS=None, feature_num = 24):
        super().__init__()
        
        self.BertCLS = BertCLS        
        
        in_features = BertCLS.classifier.in_features
        out_features = BertCLS.classifier.out_features       
        self.classifier = nn.Linear(in_features + feature_num, out_features)           
        self.classifier.weight.data.normal_(mean = 0.0, std = self.BertCLS.config.initializer_range)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()
        
        # Here you should initialize the weights of self.classifier  or USE the default initialization.
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        
        min_score = None,
        max_score = None,     
        score = None,
        
        feats = None
        
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.BertCLS.config.use_return_dict

        outputs = self.BertCLS.bert(
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

        pooled_output = outputs[1]
        
        """
        convert features into py tensor and concat with pooled_output and become a new tensor
        
        pooled_output = pooled_output + tensor(features)
        
        
        print('********shape of pooled_output**********')
        print(pooled_output.shape)
        print('***{}***'.format(type(pooled_output)))
        
        print('********shape of feats**********')
        print(feats.shape)
        print('***{}***'.format(type(feats)))
        
        pooled_output = torch.cat((pooled_output, feats),dim=1)
        
        print('********shape of pooled_output**********')
        print(pooled_output.shape)
        print('***{}***'.format(type(pooled_output)))
        
        """       
        
        pooled_output = torch.cat((pooled_output, feats),dim=1)        

        pooled_output = self.BertCLS.dropout(pooled_output) 
        
        logits = torch.sigmoid(   self.classifier(pooled_output)  )

        loss = None
        if labels is not None:
            if self.BertCLS.config.problem_type is None:
                if self.BertCLS.num_labels == 1:
                    self.BertCLS.config.problem_type = "regression"
                elif self.BertCLS.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.BertCLS.config.problem_type = "single_label_classification"
                else:
                    self.BertCLS.config.problem_type = "multi_label_classification"

            if self.BertCLS.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.BertCLS.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.BertCLS.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.BertCLS.num_labels), labels.view(-1))
            elif self.BertCLS.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
        
        
        
        
class BertWithoutFeatures(nn.Module):
    def __init__(self, BertCLS=None):
        super().__init__()
        
        self.BertCLS = BertCLS        
        
        in_features = BertCLS.classifier.in_features
        out_features = BertCLS.classifier.out_features       
        self.classifier = nn.Linear(in_features, out_features)           
        self.classifier.weight.data.normal_(mean = 0.0, std = self.BertCLS.config.initializer_range)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()
        
        # Here you should initialize the weights of self.classifier  or USE the default initialization.
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        
        min_score = None,
        max_score = None,     
        score = None,
        
        feats = None
        
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.BertCLS.config.use_return_dict

        outputs = self.BertCLS.bert(
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

        pooled_output = outputs[1]
        
        """
        convert features into py tensor and concat with pooled_output and become a new tensor
        
        pooled_output = pooled_output + tensor(features)
        
        
        print('********shape of pooled_output**********')
        print(pooled_output.shape)
        print('***{}***'.format(type(pooled_output)))
        
        print('********shape of feats**********')
        print(feats.shape)
        print('***{}***'.format(type(feats)))
        
        pooled_output = torch.cat((pooled_output, feats),dim=1)
        
        print('********shape of pooled_output**********')
        print(pooled_output.shape)
        print('***{}***'.format(type(pooled_output)))
        
        """       
        
        #pooled_output = torch.cat((pooled_output, feats),dim=1)        

        pooled_output = self.BertCLS.dropout(pooled_output) 
        
        logits = torch.sigmoid(   self.classifier(pooled_output)  )

        loss = None
        if labels is not None:
            if self.BertCLS.config.problem_type is None:
                if self.BertCLS.num_labels == 1:
                    self.BertCLS.config.problem_type = "regression"
                elif self.BertCLS.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.BertCLS.config.problem_type = "single_label_classification"
                else:
                    self.BertCLS.config.problem_type = "multi_label_classification"

            if self.BertCLS.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.BertCLS.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.BertCLS.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.BertCLS.num_labels), labels.view(-1))
            elif self.BertCLS.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
