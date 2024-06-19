import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AlbertTokenizer, AlbertModel, AutoModelForMaskedLM
from transformers import RobertaModel,BertTokenizer, RobertaConfig
from transformers import AutoConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cumsoftmax(x):
    return torch.cumsum(F.softmax(x,-1),dim=-1)

class LinearDropConnect(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, dropout=0.):
        super(LinearDropConnect, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias
        )
        self.dropout = dropout

    def sample_mask(self):
        if self.dropout == 0.:
            self._weight = self.weight
        else:
            mask = self.weight.new_empty(
                self.weight.size(),
                dtype=torch.bool
            )
            mask.bernoulli_(self.dropout)
            self._weight = self.weight.masked_fill(mask, 0.)

    def forward(self, input, sample_mask=False):
        if self.training:
            if sample_mask:
                self.sample_mask()
            return F.linear(input, self._weight, self.bias)
        else:
            return F.linear(input, self.weight * (1 - self.dropout),
                            self.bias)



class pfn_unit(nn.Module):
    def __init__(self, args, input_size):
        super(pfn_unit, self).__init__()
        self.args = args

        self.hidden_transform = LinearDropConnect(args.hidden_size, 5 * args.hidden_size, bias=True, dropout= args.dropconnect)
        self.input_transform = nn.Linear(input_size, 5 * args.hidden_size, bias=True)

        self.transform = nn.Linear(args.hidden_size*3, args.hidden_size)
        self.layer_norm = nn.LayerNorm(args.hidden_size)
        self.drop_weight_modules = [self.hidden_transform]


    def sample_masks(self):
        for m in self.drop_weight_modules:
            m.sample_mask()


    def forward(self, x, hidden):
        h_in, c_in = hidden

        gates = self.input_transform(x) + self.hidden_transform(h_in)
        c, eg_cin, rg_cin, eg_c, rg_c = gates[:, :].chunk(5, 1)

        eg_cin = 1 - cumsoftmax(eg_cin)
        rg_cin = cumsoftmax(rg_cin)

        eg_c = 1 - cumsoftmax(eg_c)
        rg_c = cumsoftmax(rg_c)

        c = torch.tanh(c)

        overlap_c = rg_c * eg_c
        upper_c = rg_c - overlap_c
        downer_c = eg_c - overlap_c

        overlap_cin =rg_cin * eg_cin
        upper_cin = rg_cin - overlap_cin
        downer_cin = eg_cin - overlap_cin

        share = overlap_cin * c_in + overlap_c * c

        c_re = upper_cin * c_in + upper_c * c + share
        c_ner = downer_cin * c_in + downer_c * c + share
        c_share = share

        h_re = torch.tanh(c_re)
        h_ner = torch.tanh(c_ner)
        h_share = torch.tanh(c_share)

        
        c_out = torch.cat((c_re, c_ner, c_share), dim=-1)
        c_out = self.transform(c_out)
        h_out = torch.tanh(c_out)

        return (h_out, c_out), (h_ner, h_re, h_share)
class RelativeMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_len):
        super(RelativeMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_len = max_len
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.relative_positions = nn.Parameter(torch.randn(num_heads, max_len, self.head_dim))

    def forward(self, query, key, value, key_padding_mask=None):
        batch_size, target_len, embed_dim = query.size()
        source_len = key.size(1)

        query = self.q_proj(query).view(batch_size, target_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(key).view(batch_size, source_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(value).view(batch_size, source_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.einsum("bnqd,bnkd->bnqk", query, key)

        # Add relative position encoding
        relative_positions = self.relative_positions[:, :source_len]
        rel_scores = torch.einsum("bnqd,nkd->bnqk", query, relative_positions)
        scores = scores + rel_scores

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.einsum("bnqk,bnvd->bnqd", attn_weights, value).transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, target_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output

# Example usage in the encoder
class encoder(nn.Module):
    def __init__(self, args, input_size):
        super(encoder, self).__init__()
        self.args = args
        self.unit = pfn_unit(args, input_size)
        self.attention = RelativeMultiheadAttention(embed_dim=args.hidden_size, num_heads=args.num_heads, max_len=args.max_position_embeddings)
        self.layer_norm = nn.LayerNorm(args.hidden_size)
        self.dropout = nn.Dropout(args.dropout)

    def hidden_init(self, batch_size):
        h0 = torch.zeros(batch_size, self.args.hidden_size).requires_grad_(False).to(device)
        c0 = torch.zeros(batch_size, self.args.hidden_size).requires_grad_(False).to(device)
        return (h0, c0)

    def apply_attention(self, h, attention_layer):
        seq_len, batch_size, _ = h.size()
        h = h.transpose(0, 1)  # (seq_len, batch_size, hidden_size)
        h_attn, _ = attention_layer(h, h, h)
        h_attn = self.dropout(h_attn)
        h_attn = self.layer_norm(h + h_attn)
        h = h_attn.transpose(0, 1)  # (batch_size, seq_len, hidden_size)
        return h


    def forward(self, x):
        seq_len = x.size(0)
        batch_size = x.size(1)
        hidden = self.hidden_init(batch_size)
        h_ner, h_re, h_share = [], [], []
        if self.training:
            self.unit.sample_masks()
        for t in range(seq_len):
            hidden, h_task = self.unit(x[t, :, :], hidden)
            h_ner.append(h_task[0])
            h_re.append(h_task[1])
            h_share.append(h_task[2])
        h_ner = torch.stack(h_ner, dim=0)
        h_re = torch.stack(h_re, dim=0)
        h_share = torch.stack(h_share, dim=0)
        h_ner = self.apply_attention(h_ner, self.attention_ner)
        h_re = self.apply_attention(h_re, self.attention_re)
        h_share = self.apply_attention(h_re, self.attention_share)
        return h_ner, h_re, h_share

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (length, batch_size, hidden_size)
        scores = self.attention(x)  # (length, batch_size, 1)
        scores = F.softmax(scores, dim=0)  # (length, batch_size, 1)
        weighted_sum = torch.sum(scores * x, dim=0)  # (batch_size, hidden_size)
        return weighted_sum
class MLPGlobalFeature(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
            super(MLPGlobalFeature, self).__init__()
            layers = []
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_size, output_size))
            self.mlp = nn.Sequential(*layers)

        def forward(self, x):
            return self.mlp(x)
class ner_unit(nn.Module):
    def __init__(self, args, ner2idx):
        super(ner_unit, self).__init__()
        self.hidden_size = args.hidden_size
        self.ner2idx = ner2idx

        self.hid2hid = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.hid2tag = nn.Linear(self.hidden_size, len(ner2idx))

        self.elu = nn.ELU()
        self.n = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.ln = nn.LayerNorm(self.hidden_size)

        self.dropout = nn.Dropout(args.dropout)
        # 引入全局特征计算的 MLP
        self.global_feature_extractor = MLPGlobalFeature(
            input_size=self.hidden_size * 2,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            num_layers=3,
            dropout=args.dropout
        )
        # 注意力池化
        self.attention_pooling = AttentionPooling(self.hidden_size)
    def compute_dynamic_threshold(self, ner):
        """
        计算基于置信度的动态阈值
        """
        # 计算平均置信度
        avg_confidence = ner.mean(dim=-1, keepdim=True)

        # 动态调整阈值，可以根据需求调整策略
        dynamic_threshold = avg_confidence * 0.5  # 例如：置信度的一半作为阈值

        return dynamic_threshold


    def forward(self, h_ner, h_share, mask):
        length, batch_size, _ = h_ner.size()

        # 计算全局特征
        h_global_input = torch.cat((h_share, h_ner), dim=-1)
        h_global_input = h_global_input.view(-1, self.hidden_size * 2)  # (length * batch_size, hidden_size * 2)
        h_global = self.global_feature_extractor(h_global_input)
        h_global = h_global.view(length, batch_size, self.hidden_size)  # (length, batch_size, hidden_size)
       # h_global = torch.max(h_global, dim=0)[0]  # (batch_size, hidden_size)
        h_global = self.attention_pooling(h_global)  # (batch_size, hidden_size)
        h_global = h_global.unsqueeze(0).repeat(length, 1, 1)
        h_global = h_global.unsqueeze(0).repeat(length, 1, 1, 1)

        st = h_ner.unsqueeze(1).repeat(1, length, 1, 1)
        en = h_ner.unsqueeze(0).repeat(length, 1, 1, 1)

        ner = torch.cat((st, en, h_global), dim=-1)
        
        
        ner = self.ln(self.hid2hid(ner))
        ner = self.elu(self.dropout(ner))
        ner = torch.sigmoid(self.hid2tag(ner))
        confidence_threshold = self.compute_dynamic_threshold(ner)

        # 应用动态阈值
        high_confidence_mask = ner > confidence_threshold

        diagonal_mask = torch.triu(torch.ones(batch_size, length, length)).to(device)

        diagonal_mask = diagonal_mask.permute(1, 2, 0)

        mask_s = mask.unsqueeze(1).repeat(1, length, 1)
        mask_e = mask.unsqueeze(0).repeat(length, 1, 1)

        mask_ner = mask_s * mask_e
        mask = diagonal_mask * mask_ner
        mask = mask.unsqueeze(-1).repeat(1, 1, 1, len(self.ner2idx))

        ner = ner * mask* high_confidence_mask

        return ner


class re_unit(nn.Module):
    def __init__(self, args, re2idx):
        super(re_unit, self).__init__()
        self.hidden_size = args.hidden_size
        self.relation_size = len(re2idx)
        self.re2idx = re2idx

        self.hid2hid = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.hid2rel = nn.Linear(self.hidden_size, self.relation_size)

        self.elu = nn.ELU()

        self.r = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.ln = nn.LayerNorm(self.hidden_size)

        self.dropout = nn.Dropout(args.dropout)
        # 引入全局特征计算的 MLP
        self.global_feature_extractor = MLPGlobalFeature(
            input_size=self.hidden_size * 2,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            num_layers=3,
            dropout=args.dropout
        )
        # 注意力池化
        self.attention_pooling = AttentionPooling(self.hidden_size)
    def compute_dynamic_threshold(self, re):
        """
        计算基于置信度的动态阈值
        """
        # 计算平均置信度
        avg_confidence = re.mean(dim=-1, keepdim=True)

        # 动态调整阈值，可以根据需求调整策略
        dynamic_threshold = avg_confidence * 0.5  # 例如：置信度的一半作为阈值

        return dynamic_threshold

    def forward(self, h_re, h_share, mask):
        length, batch_size, _ = h_re.size()

        h_global_input = torch.cat((h_share, h_re), dim=-1)
        h_global_input = h_global_input.view(-1, self.hidden_size * 2)  # (length * batch_size, hidden_size * 2)
        h_global = self.global_feature_extractor(h_global_input)
        h_global = h_global.view(length, batch_size, self.hidden_size)  # (length, batch_size, hidden_size)
        h_global = self.attention_pooling(h_global)  # (batch_size, hidden_size)
      #  h_global = torch.max(h_global, dim=0)[0]  # (batch_size, hidden_size)
        h_global = h_global.unsqueeze(0).repeat(length, 1, 1)
        h_global = h_global.unsqueeze(0).repeat(length, 1, 1, 1)

        r1 = h_re.unsqueeze(1).repeat(1, length, 1, 1)
        r2 = h_re.unsqueeze(0).repeat(length, 1, 1, 1)

        re = torch.cat((r1, r2, h_global), dim=-1)
        
        re = self.ln(self.hid2hid(re))
        re = self.elu(self.dropout(re))
        re = torch.sigmoid(self.hid2rel(re))
        # 基于置信度的动态阈值计算
        confidence_threshold = self.compute_dynamic_threshold(re)

        # 应用动态阈值
        high_confidence_mask = re > confidence_threshold

        mask = mask.unsqueeze(-1).repeat(1, 1, self.relation_size)
        mask_e1 = mask.unsqueeze(1).repeat(1, length, 1, 1)
        mask_e2 = mask.unsqueeze(0).repeat(length, 1, 1, 1)
        mask = mask_e1 * mask_e2

        re = re * mask * high_confidence_mask

        return re


class PFN(nn.Module):
    def __init__(self, args, input_size, ner2idx, rel2idx):
        super(PFN, self).__init__()
        self.args = args
        self.feature_extractor = encoder(args, input_size)

        self.ner = ner_unit(args, ner2idx)
        self.re_head = re_unit(args, rel2idx)
        self.re_tail = re_unit(args, rel2idx)
        self.dropout = nn.Dropout(args.dropout)

        if args.embed_mode == 'albert':
            self.tokenizer = AlbertTokenizer.from_pretrained("albert-xxlarge-v1")
            self.bert = AlbertModel.from_pretrained("albert-xxlarge-v1")
        elif args.embed_mode == 'bert_cased':
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
            self.bert = AutoModel.from_pretrained("bert-base-chinese")
        elif args.embed_mode == 'roberta':
            self.tokenizer = BertTokenizer.from_pretrained("./chinese-roberta-wwm-ext")
            self.config = RobertaConfig.from_pretrained("./chinese-roberta-wwm-ext/config.json")
            self.bert = RobertaModel.from_pretrained("./chinese-roberta-wwm-ext")

    def forward(self, x, mask):

        x = self.tokenizer(x, return_tensors="pt",
                                  padding='longest',
                                  is_split_into_words=True).to(device)
        x = self.bert(**x)[0]
        x = x.transpose(0, 1)

        if self.training:
            x = self.dropout(x)

        h_ner, h_re, h_share = self.feature_extractor(x)

        ner_score = self.ner(h_ner, h_share, mask)
        re_head_score = self.re_head(h_re, h_share, mask)

        re_tail_score = self.re_tail(h_share, h_re, mask)
        return ner_score, re_head_score, re_tail_score

