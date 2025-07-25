import torch
import torch.nn as nn

class StormTransformer(nn.Module):
    def __init__(self, input_dim=25, output_dim=2, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x):
        """
        x: [batch_size, seq_len, input_dim]
        """
        x = self.input_proj(x)                          # [B, T, d_model]
        x = self.pos_encoder(x)                         # Thêm positional encoding
        x = x.transpose(0, 1)                           # [T, B, d_model] cho transformer

        x = self.transformer_encoder(x)                 # [T, B, d_model]
        x = x.transpose(0, 1)                           # [B, T, d_model]

        output = self.output_proj(x)                    # [B, T, 2]
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # chẵn
        pe[:, 1::2] = torch.cos(position * div_term)  # lẻ
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]  # [B, T, d_model]
        return self.dropout(x)