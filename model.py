import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class StormCNNTransformer(nn.Module):
    def __init__(self, cnn_out_dim=128, output_dim=2, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()

        # CNN encoder
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(27, 32, kernel_size=3, padding=1),  # [B*T, 32, 33, 31]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(2),  # Bỏ MaxPool lần 1 để giữ kích thước 33x31
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [B*T, 64, 33, 31]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # [B*T, 64, 16, 15]
            nn.Flatten(),                                # [B*T, 64*16*15 = 15360]
            nn.Linear(64 * 16 * 15, cnn_out_dim),        # [B*T, cnn_out_dim]
            nn.ReLU()
        )

        self.linear_proj = nn.Linear(cnn_out_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True,
            norm_first=True  # LayerNorm trước attention
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, output_dim)

        # Khởi tạo trọng số
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x, mask=None):
        """
        x: [B, T, 27, 33, 31]
        mask: [B, T] (optional), True cho các bước hợp lệ
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)             # [B*T, 27, 33, 31]
        cnn_feat = self.cnn_encoder(x)         # [B*T, cnn_out_dim]
        cnn_feat = cnn_feat.view(B, T, -1)     # [B, T, cnn_out_dim]

        x = self.linear_proj(cnn_feat)         # [B, T, d_model]
        x = self.pos_encoder(x)                # [B, T, d_model]

        if mask is not None:
            mask = ~mask  # Chuyển True (hợp lệ) thành False (không mask)
            mask = mask.to(torch.bool)  # Đảm bảo định dạng mask
            x = self.transformer(x, src_key_padding_mask=mask)  # Mask các bước không hợp lệ
        else:
            x = self.transformer(x)

        output = self.output_proj(x)           # [B, T, 2]
        return output

# Ví dụ sử dụng mask
def create_mask(y):
    """Tạo mask từ labels, True cho các bước hợp lệ"""
    return (y != -999).any(dim=-1)  # [B, T]

# Kiểm tra mô hình
if __name__ == "__main__":
    model = StormCNNTransformer()
    x = torch.randn(16, 20, 27, 33, 31)  # Dữ liệu mẫu
    y = torch.randn(16, 20, 2)          # Labels mẫu
    mask = create_mask(y)
    output = model(x, mask)
    print(f"Output shape: {output.shape}")  # Nên là [16, 20, 2]