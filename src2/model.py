# nova_music2/src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributions as dist
from config import Config
from transformers import AutoTokenizer

import utils

class AudioSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        d = config.n_embd
        assert d % config.n_head == 0
        self.c_attn = nn.Linear(d, 3 * d)
        self.c_proj = nn.Linear(d, d)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.dk = d // config.n_head

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).chunk(3, dim=2)
        q = q.view(B, T, self.n_head, self.dk).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.dk).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.dk))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class AudioGPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Load pretrained HuggingFace tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        # Text encoder for conditioning
        self.text_embed = nn.Embedding(self.tokenizer.vocab_size, config.n_text_embd)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.n_text_embd,
            nhead=config.n_head,
            dropout=config.text_dropout
        )
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_text_layer)
        self.text_pool = nn.Linear(config.n_text_embd, config.n_embd)

        # Audio GPT
        self.embed = nn.Linear(config.n_mels, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([AudioSelfAttention(config) for _ in range(config.n_layer)])
        self.ln = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.n_mels)

    def forward(self, x, text_tokens=None):
        if x.dim() == 4:
            x = x.squeeze(1)
        B, T, M = x.size()
        x_embed = self.embed(x)
        if text_tokens is not None:
            te = self.text_embed(text_tokens)
            te = self.text_encoder(te)
            te = te.mean(dim=1).unsqueeze(1)
            cond = te.expand(-1, T, -1)
            # Falls Dimensionen nicht übereinstimmen, kürzen
            if cond.size(-1) != x_embed.size(-1):
                cond = cond[:, :, :x_embed.size(-1)]
        else:
            cond = torch.zeros_like(x_embed)

        x = x_embed + cond + self.pos_emb[:, :T, :]
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln(x)
        return self.head(x)


    def encode_text(self, texts):
        """Encodes raw text strings into token IDs for embedding."""
        tokens = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        return tokens["input_ids"].to(next(self.parameters()).device)

    def generate(self, prior, raw_texts=None, length=None, temperature=1.0, writer=None):
        device = prior.device
        length = length or self.config.block_size
        seq = prior

        if raw_texts is not None:
            text_tokens = self.encode_text(raw_texts)
            te = self.text_embed(text_tokens)
            te = self.text_encoder(te)
            te = te.mean(dim=1).unsqueeze(1)
        else:
            te = None

        for _ in range(length):
            if seq.size(1) > self.config.block_size:
                seq = seq[:, 1:]
            B, T, _ = seq.size()
            cond = te.expand(B, T, -1) if te is not None else 0
            out = self.forward(seq, text_tokens if te is not None else None)
            logits = F.log_softmax(out[:, -1] / temperature, dim=-1)
            probs = torch.exp(logits)
            if writer:
                writer.add_histogram('probs', probs)
            next_token = dist.Categorical(probs).sample().unsqueeze(1)
            next_vec = F.one_hot(next_token, num_classes=self.config.n_mels).float()
            seq = torch.cat((seq, next_vec), dim=1)
        return seq
