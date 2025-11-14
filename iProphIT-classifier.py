import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import argparse
import os
import sys

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置内存优化选项
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(47003)

# 词汇表（独热编码）
vocab = {'A': [1, 0, 0, 0], 'G': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'T': [0, 0, 0, 1], '[PAD]': [0, 0, 0, 0]}
input_dim = 4
vocab_map = {'A': 0, 'G': 1, 'C': 2, 'T': 3, '[PAD]': 4}
vocab_array = np.array(list(vocab.values()), dtype=np.float32)

# 批量将字符串序列转换为张量表示（向量化）
def batch_str_to_tensor(seq_strs, vocab_map, vocab_array, n_threshold=0.025):
    results = []
    valid_bases = ['A', 'G', 'C', 'T']
    ambiguous_base_map = {
        'W': ['A', 'T'], 'R': ['A', 'G'], 'Y': ['C', 'T'], 'S': ['C', 'G'],
        'M': ['A', 'C'], 'K': ['G', 'T'], 'N': ['A', 'G', 'C', 'T'],
        'B': ['C', 'G', 'T'], 'D': ['A', 'G', 'T'], 'H': ['A', 'C', 'T'],
        'V': ['A', 'C', 'G']
    }
    non_agct_chars = {}
    non_agct_count = 0
    
    for seq_idx, seq_str in enumerate(seq_strs):
        seq_str = seq_str.strip().replace(' ', '').replace('\n', '').replace('\r', '').upper()
        total_length = len(seq_str)
        n_count = seq_str.count('N')
        n_ratio = n_count / total_length if total_length > 0 else 0

        if n_ratio > n_threshold:
            logger.warning(f"Sequence {seq_idx} skipped due to high N ratio: {n_ratio:.4f}")
            results.append(None)
            continue

        seq_non_agct = {}
        for base in seq_str:
            if base not in valid_bases:
                seq_non_agct[base] = seq_non_agct.get(base, 0) + 1
                non_agct_chars[base] = non_agct_chars.get(base, 0) + 1
                non_agct_count += 1

        if seq_non_agct:
            logger.info(f"Sequence {seq_idx} contains non-AGCT characters: {seq_non_agct}")

        seq_list = list(seq_str)
        np.random.seed(47003)
        for i, base in enumerate(seq_list):
            if base not in valid_bases:
                possible_bases = ambiguous_base_map.get(base, ['A', 'G', 'C', 'T'])
                seq_list[i] = np.random.choice(possible_bases)
        np.random.seed(None)
        seq_str = ''.join(seq_list)
        
        seq_array = np.array([vocab_map.get(base, vocab_map['[PAD]']) for base in seq_str], dtype=np.int32)
        one_hot = vocab_array[seq_array]
        results.append(torch.tensor(one_hot, dtype=torch.float))

    if non_agct_chars:
        logger.info(f"Total non-AGCT characters found: {non_agct_count}")
        logger.info(f"Non-AGCT characters and their counts: {non_agct_chars}")
    else:
        logger.info("No non-AGCT characters found in the sequences.")

    return results

# 预处理FASTA文件
def preprocess_fasta(fasta_file, vocab_map, vocab_array, n_threshold=0.025):
    records = list(SeqIO.parse(fasta_file, "fasta"))
    logger.info(f"Read {len(records)} sequences from {fasta_file}")
    seq_strs = [str(record.seq) for record in records]
    tensors = batch_str_to_tensor(seq_strs, vocab_map, vocab_array, n_threshold)
    valid_tensors = [t for t in tensors if t is not None and t.size(0) > 0]
    valid_records = [records[i] for i, t in enumerate(tensors) if t is not None and t.size(0) > 0]
    
    if not valid_tensors:
        raise ValueError(f"No valid sequences found in {fasta_file}")
    
    logger.info(f"Processed {len(valid_tensors)} valid sequences")
    return valid_tensors, valid_records

# 定义数据集类
class PredictionDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

# 填充函数
def prediction_collate_fn(batch):
    sequences = batch
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    original_lengths = torch.tensor([len(seq) for seq in sequences])
    return padded_sequences, original_lengths

# 加载模型（只包含必要的模型类）
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_k, max_seq_len=7000):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.theta_segments = [
            (500, 10000.0), (1000, 8000.0), (1500, 6000.0), (2000, 5000.0), (None, 4500.0)
        ]
        for threshold, base_theta in self.theta_segments:
            theta = base_theta ** (-2.0 * (torch.arange(0, d_k, 2).float()) / d_k)
            self.register_buffer(f'theta_{threshold if threshold is not None else "max"}', theta)

    def get_theta_for_length(self, seq_len):
        for threshold, _ in self.theta_segments:
            if threshold is None or seq_len <= threshold:
                return getattr(self, f'theta_{threshold if threshold is not None else "max"}')
        return getattr(self, f'theta_max')

    def forward(self, x):
        batch_size, nhead, seq_len, d_k = x.size()
        assert d_k == self.d_k
        theta = self.get_theta_for_length(seq_len)
        positions = torch.arange(seq_len, device=x.device).float().unsqueeze(1)
        angles = positions * theta
        sin_vals = torch.sin(angles).unsqueeze(0).unsqueeze(1)
        cos_vals = torch.cos(angles).unsqueeze(0).unsqueeze(1)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rot_even = x_even * cos_vals - x_odd * sin_vals
        x_rot_odd = x_even * sin_vals + x_odd * cos_vals
        x_rot = torch.zeros_like(x)
        x_rot[..., 0::2] = x_rot_even
        x_rot[..., 1::2] = x_rot_odd
        return x_rot

class RotaryMultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.3, device='cpu'):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEncoding(self.d_k)
        self.attn = None

    def forward(self, query, key, value, attn_mask=None, training=True):
        batch_size = query.size(0)
        seq_len = query.size(1)
        q = self.q_proj(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        q = self.rope(q)
        k = self.rope(k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        if attn_mask is not None:
            attn_mask = attn_mask.expand(batch_size, self.nhead, seq_len, seq_len)
            scores = scores + attn_mask
        attn = torch.softmax(scores, dim=-1)
        if torch.isnan(attn).any() or torch.isinf(attn).any():
            attn = torch.zeros_like(attn)
        self.attn = attn
        attn = self.dropout(attn) if training else attn
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(context)
        return output, attn

class RotaryTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.3, device='cpu'):
        super().__init__()
        self.self_attn = RotaryMultiheadAttention(d_model, nhead, dropout=dropout, device=device)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src, src_mask=None, training=True):
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask, training=training)
        src = src + self.dropout1(src2) if training else src + src2
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))) if training else self.activation(self.linear1(src)))
        src = src + self.dropout2(src2) if training else src + src2
        return self.norm2(src), attn

class CNNLayer(nn.Module):
    def __init__(self, input_dim=4, d_model=256, window_size=256, overlap_percent=0.25, dropout=0.5):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.window_size = window_size
        self.stride = int(window_size * (1 - overlap_percent))
        self.conv3 = nn.Sequential(
            nn.Conv1d(d_model, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(d_model, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(d_model, 128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        self.conv9 = nn.Sequential(
            nn.Conv1d(d_model, 192, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.BatchNorm1d(192),
        )
        self.res_proj3 = nn.Conv1d(d_model, 64, kernel_size=1)
        self.res_proj5 = nn.Conv1d(d_model, 128, kernel_size=1)
        self.res_proj7 = nn.Conv1d(d_model, 128, kernel_size=1)
        self.res_proj9 = nn.Conv1d(d_model, 192, kernel_size=1)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1)
        self.final_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, original_lengths, training=True):
        batch_size, seq_len, _ = x.size()
        x = self.proj(x)
        num_windows = np.ceil((seq_len - self.window_size) / self.stride).astype(int) + 1
        padded_len = (num_windows - 1) * self.stride + self.window_size
        mask = torch.ones(batch_size, num_windows, dtype=torch.bool, device=x.device)
        for i, length in enumerate(original_lengths):
            valid_windows = np.ceil((length.item() - self.window_size) / self.stride).astype(int) + 1
            if valid_windows < num_windows:
                mask[i, valid_windows:] = False
        if seq_len < padded_len:
            padding = torch.zeros(batch_size, padded_len - seq_len, x.size(-1), device=x.device)
            x = torch.cat([x, padding], dim=1)
        x = x.unfold(1, self.window_size, self.stride)
        x = x.reshape(-1, x.size(2), x.size(3))
        residual = x
        x3 = self.pool(self.conv3(x) + self.res_proj3(residual))
        x5 = self.pool(self.conv5(x) + self.res_proj5(residual))
        x7 = self.pool(self.conv7(x) + self.res_proj7(residual))
        x9 = self.pool(self.conv9(x) + self.res_proj9(residual))
        combined = torch.cat([x3, x5, x7, x9], dim=1)
        pooled = self.final_pool(combined)
        x = pooled.view(batch_size, num_windows, -1)
        x = self.dropout(x) if training else x
        return x, mask

class FullyConnectedLayer(nn.Module):
    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.residual_proj1 = nn.Linear(d_model, 512)
        self.residual_proj2 = nn.Linear(512, 256)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(256)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res1 = self.residual_proj1(x)
        x = self.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.norm1(x + res1)
        res2 = self.residual_proj2(x)
        x = self.gelu(self.fc2(x))
        x = self.dropout(x)
        x = self.norm2(x + res2)
        x = self.fc3(x)
        return x

class TransformerLayer(nn.Module):
    def __init__(self, d_model=512, nhead=16, num_layers=2, dropout=0.3, cls_dropout=0.1, device='cpu'):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.transformer_layers = nn.ModuleList([
            RotaryTransformerEncoderLayer(d_model, nhead, d_model*4, dropout, device)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.nhead = nhead

    def create_padding_mask(self, mask, device):
        batch_size, num_windows = mask.size()
        seq_len = num_windows + 1
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        extended_mask = torch.cat([cls_mask, mask], dim=1)
        attn_mask = torch.zeros(batch_size, self.nhead, seq_len, seq_len, device=device)
        invalid_mask = ~extended_mask.unsqueeze(-1) * ~extended_mask.unsqueeze(-2)
        attn_mask.masked_fill_(invalid_mask.unsqueeze(1), float('-inf'))
        return attn_mask

    def forward(self, x, mask, original_lengths, training=True):
        batch_size = x.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        attn_mask = self.create_padding_mask(mask, x.device)
        attn_weights = None
        for layer in self.transformer_layers:
            x, attn = layer(x, src_mask=attn_mask, training=training)
            attn_weights = attn
        cls_output = self.norm(x[:, 0, :])
        return cls_output, attn_weights

class SiameseCNNTransformerModel(nn.Module):
    def __init__(self, input_dim=4, d_model=256, nhead=16, num_layers=2, window_size=256, overlap_percent=0.25, 
                 cnn_dropout=0.5, attn_dropout=0.3, ffn_dropout=0.2, cls_dropout=0.1, device='cpu'):
        super().__init__()
        self.cnn_layer = CNNLayer(input_dim, d_model, window_size, overlap_percent, cnn_dropout)
        self.transformer_layer = TransformerLayer(d_model=512, nhead=nhead, num_layers=num_layers, dropout=attn_dropout, cls_dropout=cls_dropout, device=device)
        self.classifier = FullyConnectedLayer(d_model=512, dropout=cls_dropout)
        self.siamese_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )

    def forward(self, x, original_lengths, training=True, return_features=False):
        cnn_out, mask = self.cnn_layer(x, original_lengths, training=training)
        transformer_out, attn_weights = self.transformer_layer(cnn_out, mask, original_lengths, training=training)
        if return_features:
            features = self.siamese_head(transformer_out)
            return features, attn_weights
        logits = self.classifier(transformer_out)
        return logits, attn_weights

# 预测函数（输出 active/dormant）
def predict(model, dataloader, records, device, output_file):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i, (inputs, original_lengths) in enumerate(dataloader):
            inputs, original_lengths = inputs.to(device), original_lengths.to(device)
            outputs, _ = model(inputs, original_lengths, training=False)
            _, predicted = torch.max(outputs, 1)
            
            start_idx = i * dataloader.batch_size
            end_idx = min(start_idx + dataloader.batch_size, len(records))
            batch_records = records[start_idx:end_idx]
            
            for j in range(len(predicted)):
                if j < len(batch_records):
                    label = "active" if predicted[j].item() == 1 else "dormant"
                    predictions.append(f"{batch_records[j].id}\t{label}")
                else:
                    logger.warning(f"Index {start_idx + j} out of bounds for records")
            
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("ID\tPredict\n")
        f.write("\n".join(predictions))
    logger.info(f"Predictions saved to {output_file}")

def main(model_path, fasta_file, output_file, batch_size=4, window_size=128, overlap_percent=0.25, num_threads=32):
    torch.set_num_threads(num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 加载模型
    model = SiameseCNNTransformerModel(
        input_dim=4, d_model=256, nhead=16, num_layers=2, window_size=window_size,
        overlap_percent=overlap_percent, cnn_dropout=0.2, attn_dropout=0.3, ffn_dropout=0.2, cls_dropout=0.5, device=device
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    logger.info(f"Loaded model from {model_path}")

    # 预处理FASTA文件
    sequences, records = preprocess_fasta(fasta_file, vocab_map, vocab_array)
    
    # 创建数据集和数据加载器
    dataset = PredictionDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                           collate_fn=prediction_collate_fn, num_workers=0)

    # 进行预测
    predict(model, dataloader, records, device, output_file)
    logger.info("Prediction completed")


if __name__ == "__main__":
    logo = r"""
          _ ____                  _     ___ _____ 
         (_)  _ \ _ __ ___  _ __ | |__ |_ _|_   _|
         | | |_) | '__/ _ \| '_ \| '_ \ | |  | |  
         | |  __/| | | (_) | |_) | | | || |  | |  
         |_|_|   |_|  \___/| .__/|_| |_|___| |_|  
                           |_|                    v1.0    
                           
        """

    print(logo)
    # 先创建 parser
    parser = argparse.ArgumentParser(
         formatter_class=argparse.RawTextHelpFormatter
    )

    # 再添加参数
    parser.add_argument('-i', '--input', required=True, help='Path to the input FASTA file (required)')
    parser.add_argument('-m', '--model', default='./iProphIT_model-v1.pth', help='Path to the trained model file (default: ./iProphIT_model-v1.pth)')
    parser.add_argument('-o', '--output', default='./Result.tsv', help='Output TSV file path (default: ./Result.tsv)')
    parser.add_argument('-t', '--threads', type=int, default=4, help='Number of CPU threads for DataLoader (default: 4)')

    args = parser.parse_args()

    # 检查输出扩展名
    if not args.output.lower().endswith('.tsv'):
        print("Error: Output file must have .tsv extension.", file=sys.stderr)
        sys.exit(1)

    # 检查输入文件存在
    if not os.path.isfile(args.input):
        print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    # 检查模型文件存在
    if not os.path.isfile(args.model):
        print(f"Error: Model file '{args.model}' not found.", file=sys.stderr)
        sys.exit(1)

    # 调用主函数
    main(
        model_path=args.model,
        fasta_file=args.input,
        output_file=args.output,
        batch_size=4,                 # 固定 batch_size 为 4
        window_size=128,              # 固定 window_size 为 128
        overlap_percent=0.25,         # 固定 overlap_percent 为 0.25
        num_threads=args.threads
    )