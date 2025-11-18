import torch
import torch.nn as nn


def generate_square_subsequent_mask(sz: int):
    """Causal mask to stop attention to future positions"""
    return torch.triu(torch.ones(sz, sz), diagonal=1).bool()


class SketchTransformer(nn.Module):
    def __init__(
        self, vocab_size, d_model=256, nhead=8, num_layers=6, max_len=200, **kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_len = max_len

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model
        )  # , activation='gelu'
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Register buffers
        self.register_buffer("causal_mask", generate_square_subsequent_mask(max_len))
        self.register_buffer("positions", torch.arange(max_len).unsqueeze(0))

    def forward(self, x, *args, src_key_padding_mask=None):
        """
        x: (batch, seq_len) input tokens
        Returns: (batch, seq_len, vocab_size) logits
        """
        batch_size, seq_len = x.shape
        positions = self.positions[:, :seq_len]
        x = self.embed(x) + self.pos_embed(positions)  # (batch, seq_len, d_model)
        x = x.transpose(0, 1)  # -> (seq_len, batch, d_model)
        mask = self.causal_mask[:seq_len, :seq_len]
        x = self.transformer(
            x, mask=mask, src_key_padding_mask=src_key_padding_mask
        )  # (seq_len, batch, d_model)
        x = x.transpose(0, 1)  # back to (batch, seq_len, d_model)
        logits = self.fc_out(x)  # (batch, seq_len, vocab_size)
        return logits


class SketchTransformerConditional(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes,
        d_model=512,
        nhead=8,
        num_layers=6,
        max_len=200,
        **kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_len = max_len
        self.num_classes = num_classes

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.class_embed = nn.Embedding(num_classes, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Register buffers
        self.register_buffer("causal_mask", generate_square_subsequent_mask(max_len))
        self.register_buffer("positions", torch.arange(max_len).unsqueeze(0))

    def forward(self, x, class_labels, src_key_padding_mask=None):
        """
        x: (batch, seq_len) input tokens
        class_labels: (batch,) integer labels for conditioning
        Returns: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        positions = self.positions[:, :seq_len]
        x = self.embed(x) + self.pos_embed(positions)
        class_cond = self.class_embed(class_labels).unsqueeze(1)  # (batch, 1, d_model)
        x = x + class_cond  # simple additive conditioning
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        mask = self.causal_mask[:seq_len, :seq_len]
        x = self.transformer(x, mask=mask, src_key_padding_mask=src_key_padding_mask)
        x = x.transpose(0, 1)  # back to (batch, seq_len, d_model)
        logits = self.fc_out(x)
        return logits
