# nova_music2/src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributions as dist
from config import Config
import utils

# Selbst-Attention Layer für Audio GPT
class AudioSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).chunk(3, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y

class AudioGPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embedding = nn.Linear(config.input_vocab_size, config.n_embd)
        self.pos_embedding = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([AudioSelfAttention(config) for _ in range(config.n_layer)])
        self.norm = nn.LayerNorm(config.n_embd)
        self.output = nn.Linear(config.n_embd, config.input_vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x + self.pos_embedding[:, :x.size(-3), :]
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.output(x)
        return x

    def generate(self, prior: torch.Tensor, length: int = 2048, tf_board_writer=None, temperature: float = 1.2) -> torch.Tensor:
        """
        Generates new audio sequences based on prior input.
        Args:
            prior (torch.Tensor): The input tensor (e.g., a seed Mel-spectrogram).
            length (int): The number of steps to generate.
            tf_board_writer (SummaryWriter, optional): A TensorBoard writer for logging.
            temperature (float): Controls the randomness of sampling (higher values make the output more random).
        
        Returns:
            torch.Tensor: The generated sequence.
        """
        decode_array = prior
        result_array = prior

        for i in range(length):
            if decode_array.size(1) >= self.config.block_size:
                decode_array = decode_array[:, 1:]  # Truncate to keep the sequence length manageable.

            # Create the look-ahead mask for the decoder
            padding_mask, look_ahead_mask = utils.get_masked_with_pad_tensor(
                decode_array.size(1), decode_array, decode_array, pad_token=self.config.pad_token
            )

            # Forward pass through the decoder layers
            result = decode_array
            for layer in self.layers:
                result = layer(result)  # Apply self-attention layers sequentially
            
            result = self.output(result)
            result = result.softmax(-1)  # Get probabilities from the logits

            # Optionally log the results to TensorBoard
            if tf_board_writer:
                tf_board_writer.add_image("logits", result, global_step=i)

            # Sampling based on the probabilities
            if temperature != 1.0:
                result = result / temperature  # Apply temperature scaling

            # Choose the next token (or Mel-spectrogram frame)
            if torch.rand(1).item() > 0.5:  # Use sampling with probability
                pdf = dist.OneHotCategorical(probs=result[:, -1])
                sampled_token = pdf.sample().unsqueeze(-1)
                decode_array = torch.cat((decode_array, sampled_token), dim=-1)
            else:
                # Greedy sampling (choose the most probable token)
                sampled_token = result[:, -1].argmax(-1).unsqueeze(-1)
                decode_array = torch.cat((decode_array, sampled_token), dim=-1)

            # Accumulate the generated result
            result_array = torch.cat((result_array, sampled_token), dim=-1)

            # Clean up the mask
            del look_ahead_mask

        # Return the final generated sequence
        result_array = result_array[0]  # We assume batch size is 1
        return result_array
