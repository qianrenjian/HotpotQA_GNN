import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLNetPreTrainedModel, XLNetModel
from transformers import PreTrainedModel
from transformers import AutoConfig, AutoModel
from copy import deepcopy

# modified from original modeling_xlnet.py.

class PoolerStartLogits_GRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.BiGRU = nn.GRU(input_size=config.hidden_size,
                                hidden_size=config.hidden_size, 
                                num_layers=2, 
                                batch_first=True, 
                                dropout=0.1, 
                                bidirectional=True)
        self.activation = F.leaky_relu
        self.dense_0 = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dense_1 = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, p_mask=None):
        """ Args:
            **p_mask**: (`optional`) ``torch.FloatTensor`` of shape `(batch_size, seq_len)`
                invalid position mask such as query and special symbols (PAD, SEP, CLS)
                1.0 means token should be masked.
        """
        x = self.activation(self.BiGRU(hidden_states)[0])
        x = self.activation(hidden_states + self.dense_0(x))
        x = self.dense_1(x).squeeze(-1)

        if p_mask is not None:
            if next(self.parameters()).dtype == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x

class PoolerEndLogits_GRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.BiGRU = nn.GRU(input_size=config.hidden_size,
                                hidden_size=config.hidden_size, 
                                num_layers=2, 
                                batch_first=True, 
                                dropout=0.1, 
                                bidirectional=True)
        self.activation = F.leaky_relu
        if 'layer_norm_eps' not in config.to_dict():
            layer_norm_eps = 1e-12
        else:
            layer_norm_eps = config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size*2, eps=layer_norm_eps)
        self.dense_1 = nn.Linear(config.hidden_size*2, 1)

    def forward(self, hidden_states, start_states=None, start_positions=None, p_mask=None, ignore_index=-100):
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            start_positions = deepcopy(start_positions)
            start_positions -= start_positions.eq(ignore_index)*ignore_index
            slen, hsz = hidden_states.shape[-2:]
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions)  # shape (bsz, 1, hsz)
            start_states = start_states.expand(-1, slen, -1)  # shape (bsz, slen, hsz)

        x0 = torch.cat([hidden_states, start_states], dim=-1)
        x1 = self.BiGRU(self.dense_0(x0))[0]
        x = self.activation(x0 + x1)
        x = self.LayerNorm(x)
        x = self.dense_1(x).squeeze(-1)
        del x0, x1

        if p_mask is not None:
            if next(self.parameters()).dtype == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x

class PoolerStartLogits_Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer_encoder = nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=8),
                        num_layers=1)
        self.activation = F.leaky_relu
        self.dense_1 = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, p_mask=None):
        x = self.activation(self.transformer_encoder(hidden_states.transpose(0,1)))
        x = self.dense_1(x.transpose(0,1)).squeeze(-1)

        if p_mask is not None:
            if next(self.parameters()).dtype == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x

class PoolerEndLogits_Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=8),
                        num_layers=1)
        self.activation = F.leaky_relu
        if 'layer_norm_eps' not in config.to_dict():
            layer_norm_eps = 1e-12
        else:
            layer_norm_eps = config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=layer_norm_eps)
        self.dense_1 = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, start_states=None, start_positions=None, p_mask=None, ignore_index=-100):
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            start_positions = deepcopy(start_positions)
            start_positions -= start_positions.eq(ignore_index)*ignore_index
            slen, hsz = hidden_states.shape[-2:]
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions)  # shape (bsz, 1, hsz)
            start_states = start_states.expand(-1, slen, -1)  # shape (bsz, slen, hsz)

        x = self.dense_0(torch.cat([hidden_states, start_states], dim=-1))
        x = self.transformer_encoder(x.transpose(0,1))
        x = self.activation(x.transpose(0,1))
        x = self.LayerNorm(x)
        x = self.dense_1(x).squeeze(-1)

        if p_mask is not None:
            if next(self.parameters()).dtype == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x

class PoolerStartLogits(nn.Module):
    """ Compute SQuAD start_logits from sequence hidden states. """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, p_mask=None):
        """ Args:
            **p_mask**: (`optional`) ``torch.FloatTensor`` of shape `(batch_size, seq_len)`
                invalid position mask such as query and special symbols (PAD, SEP, CLS)
                1.0 means token should be masked.
        """
        x = self.dense(hidden_states).squeeze(-1)

        if p_mask is not None:
            if next(self.parameters()).dtype == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x

class PoolerEndLogits(nn.Module):
    """ Compute SQuAD end_logits from sequence hidden states and start token hidden state.
    """

    def __init__(self, config):
        super().__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = F.leaky_relu
        if 'layer_norm_eps' not in config.to_dict():
            layer_norm_eps = 1e-12
        else:
            layer_norm_eps = config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=layer_norm_eps)
        self.dense_1 = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, start_states=None, start_positions=None, p_mask=None, ignore_index=-100):
        """ Args:
            One of ``start_states``, ``start_positions`` should be not None.
            If both are set, ``start_positions`` overrides ``start_states``.

            **start_states**: ``torch.LongTensor`` of shape identical to hidden_states
                hidden states of the first tokens for the labeled span.
            **start_positions**: ``torch.LongTensor`` of shape ``(batch_size,)``
                position of the first token for the labeled span:
            **p_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, seq_len)``
                Mask of invalid position such as query and special symbols (PAD, SEP, CLS)
                1.0 means token should be masked.
        """
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            start_positions = deepcopy(start_positions)
            start_positions -= start_positions.eq(ignore_index)*ignore_index
            slen, hsz = hidden_states.shape[-2:]
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions)  # shape (bsz, 1, hsz)
            start_states = start_states.expand(-1, slen, -1)  # shape (bsz, slen, hsz)

        x = self.dense_0(torch.cat([hidden_states, start_states], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x).squeeze(-1)

        if p_mask is not None:
            if next(self.parameters()).dtype == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x

class PoolerAnswerClass(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation = F.leaky_relu
        self.dense = nn.Linear(config.hidden_size, 2)

    def forward(self, hidden_states, cls_index=None):
        hsz = hidden_states.shape[-1]

        if cls_index is not None:
            cls_index = cls_index[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            cls_token_state = hidden_states.gather(-2, cls_index).squeeze(-2)  # shape (bsz, hsz)
        else:
            cls_token_state = hidden_states[:, -1, :]  # shape (bsz, hsz)

        x = self.dense(cls_token_state)

        return x

class AutoQuestionAnswering(PreTrainedModel):
    def __init__(self, config, LM_Model):
        super().__init__(config)
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top

        self.transformer = LM_Model
        self.start_logits = PoolerStartLogits_GRU(config)
        self.end_logits = PoolerEndLogits_GRU(config)
        self.answer_class = PoolerAnswerClass(config)

        self.cls_index=config.cls_index

        self.init_weights()

    def init_weights(self):
        # """ Initialize the weights.
        # """
        # if isinstance(module, (nn.Linear, nn.Embedding)):
        #     module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        #     if isinstance(module, (nn.Linear)) and module.bias is not None:
        #         module.bias.data.zero_()
        # elif isinstance(module, nn.LayerNorm):
        #     module.bias.data.zero_()
        #     module.weight.data.fill_(1.0)
        for model in [self.start_logits, self.end_logits, self.answer_class]:
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.kaiming_normal_(p)

    def freeze_to_layer_by_name(self, layer_name):
        '''冻结层. 从0到layer_name.'''
        if layer_name == None: return
        if layer_name == 'all':
            index_start = len(self.transformer.state_dict())
        else:
            index_start = -1
            for index, (key, _value) in enumerate(self.transformer.state_dict().items()):
                if layer_name in key: 
                    index_start = index
                    break

        if index_start < 0:
            print(f"Don't find layer name: {layer_name}")
            print(f"must in : \n{self.transformer.state_dict().keys()}")
            return
        
        no_grad_nums = index_start + 1
        grad_nums = 0

        for index, i in enumerate(self.transformer.parameters()):
            if index >= index_start:
                i.requires_grad = True
                grad_nums += 1
            else:
                i.requires_grad = False
        print(f"freeze layers num: {no_grad_nums}, active layers num: {grad_nums}.")
        # no need to return.
    
    @classmethod
    def from_pretrained(cls, model_path, cls_index=0):
        model = AutoModel.from_pretrained(model_path, local_files_only=True)
        config = model.config
        config_dict = config.to_dict()
        config_dict['start_n_top'] = 5
        config_dict['end_n_top'] = 5
        config_dict['cls_index'] = cls_index
        config = config.from_dict(config_dict)
        return cls(config, model)
    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            special_tokens_mask=None,
        ):

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            # position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            )
        hidden_states = transformer_outputs[0]
        print(f"hidden_states: {hidden_states.device}")
        start_logits = self.start_logits(hidden_states, p_mask=special_tokens_mask)

        outputs = transformer_outputs[1:]  # Keep mems, hidden states, attentions if there are in it
        bsz = start_positions.shape[0]
        cls_index = torch.zeros([bsz], device = input_ids.device).fill_(self.cls_index).long()
        if (start_positions is not None and end_positions is not None):
            # If we are on multi-GPU, let's remove the dimension added by batch splitting
            for x in (start_positions, end_positions, cls_index):
                if x is not None and x.dim() > 1:
                    x.squeeze_(-1)

            # during training, compute the end logits based on the ground truth of the start position
            end_logits = self.end_logits(hidden_states, start_positions=start_positions, p_mask=special_tokens_mask)
            cls_logits = self.answer_class(hidden_states, cls_index=cls_index)
            outputs = (start_logits, end_logits, cls_logits) + outputs

            return outputs

        else:
            # during inference, compute the end logits based on beam search
            bsz, slen, hsz = hidden_states.size()
            start_log_probs = F.softmax(start_logits, dim=-1)  # shape (bsz, slen)

            start_top_log_probs, start_top_index = torch.topk(
                start_log_probs, self.start_n_top, dim=-1
            )  # shape (bsz, start_n_top)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)  # shape (bsz, start_n_top, hsz)
            start_states = torch.gather(hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)  # shape (bsz, slen, start_n_top, hsz)

            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(
                start_states
            )  # shape (bsz, slen, start_n_top, hsz)
            special_tokens_mask = special_tokens_mask.unsqueeze(-1) if special_tokens_mask is not None else None
            end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=special_tokens_mask)
            end_log_probs = F.softmax(end_logits, dim=1)  # shape (bsz, slen, start_n_top)

            end_top_log_probs, end_top_index = torch.topk(
                end_log_probs, self.end_n_top, dim=1
            )  # shape (bsz, end_n_top, start_n_top)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

            start_states = torch.einsum(
                "blh,bl->bh", hidden_states, start_log_probs
            )  # get the representation of START as weighted sum of hidden states
            cls_logits = self.answer_class(
                hidden_states, start_states=start_states, cls_index=cls_index
            )  # Shape (batch size,): one single `cls_logits` for each sample

            outputs = (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits) + outputs

            return outputs











