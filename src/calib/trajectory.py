"""B6 / B7: a small transformer over the token confidence trajectory.

Each generation's per-token log probabilities are compressed into 256 chunk summaries with 4 channels
(mean, min, share below -2, share of tokens inside the code span), plus log of the token count."""

import numpy as np
import torch
from torch import nn

N_CHUNKS = 256


def chunk_sequence(lp, start, end, n_chunks=N_CHUNKS):
    """(n_chunks, 4) summaries and a padding mask. Short outputs get one token per chunk, then padding."""
    n = len(lp)
    in_code = np.zeros(n, bool)
    if end > start:
        in_code[start:end] = True
    out = np.zeros((n_chunks, 4), np.float32)
    pad = np.ones(n_chunks, bool)
    pieces = np.array_split(np.arange(n), n_chunks) if n >= n_chunks else [np.array([i]) for i in range(n)]
    for c, idx in enumerate(pieces):
        seg = lp[idx]
        out[c] = [seg.mean(), seg.min(), (seg < -2).mean(), in_code[idx].mean()]
        pad[c] = False
    return out, pad


def build_tensors(rows):
    """Chunk arrays (n, 256, 4), padding masks (n, 256), and log token counts (n,) for every row."""
    seqs, pads = zip(*[chunk_sequence(lp, s, e) for lp, s, e in zip(rows.logprobs, rows.code_start, rows.code_end)])
    return np.stack(seqs), np.stack(pads), np.log(rows.logprobs.map(len).to_numpy().astype(np.float32))


class Standardizer:
    """Channel means and stds from base_train rows only, over real (unpadded) chunks."""

    def fit(self, seq, pad, scalar, extra=None):
        real = seq[~pad]
        self.mu, self.sd = real.mean(0), real.std(0) + 1e-6
        self.s_mu, self.s_sd = scalar.mean(), scalar.std() + 1e-6
        if extra is not None:
            self.e_mu, self.e_sd = extra.mean(0), extra.std(0) + 1e-6
        return self

    def transform(self, seq, pad, scalar, extra=None):
        seq = np.where(pad[..., None], 0.0, (seq - self.mu) / self.sd).astype(np.float32)
        scalar = ((scalar - self.s_mu) / self.s_sd).astype(np.float32)
        extra = None if extra is None else ((extra - self.e_mu) / self.e_sd).astype(np.float32)
        return seq, scalar, extra


class TrajectoryNet(nn.Module):
    def __init__(self, n_extra=0, d=64, heads=4, layers=2, dropout=0.1, n_chunks=N_CHUNKS):
        super().__init__()
        self.proj = nn.Linear(4, d)
        self.pos = nn.Parameter(torch.zeros(1, n_chunks, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(d + 1 + n_extra, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, seq, pad, scalar, extra=None):
        h = self.encoder(self.proj(seq) + self.pos, src_key_padding_mask=pad)
        keep = (~pad).unsqueeze(-1).float()
        pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1)  # mean over real chunks only
        parts = [pooled, scalar.unsqueeze(-1)] + ([extra] if extra is not None else [])
        return self.head(torch.cat(parts, -1)).squeeze(-1)


def _tensors(seq, pad, scalar, extra, idx, device="cpu"):
    t = [torch.from_numpy(seq[idx]), torch.from_numpy(pad[idx]), torch.from_numpy(scalar[idx])]
    t = t + [torch.from_numpy(extra[idx]) if extra is not None else None]
    return [x.to(device) if x is not None else None for x in t]


def predict(model, seq, pad, scalar, extra=None, batch=512):
    device = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(seq), batch):
            idx = np.arange(i, min(i + batch, len(seq)))
            out.append(torch.sigmoid(model(*_tensors(seq, pad, scalar, extra, idx, device))).cpu().numpy())
    return np.concatenate(out)


def train_one(seq, pad, scalar, extra, y, tr, va, seed, epochs=50, patience=5, batch=128, lr=1e-3, wd=1e-2,
              device="cpu"):
    """AdamW on binary cross entropy; early stop on the inner validation rows; keep the best epoch's weights.
    Returns the model on the CPU."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TrajectoryNet(n_extra=0 if extra is None else extra.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss()
    yt = torch.from_numpy(y.astype(np.float32)).to(device)
    best, best_state, bad, history = np.inf, None, 0, []
    for epoch in range(epochs):
        model.train()
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // batch)):
            opt.zero_grad()
            loss = loss_fn(model(*_tensors(seq, pad, scalar, extra, b, device)), yt[torch.from_numpy(b).to(device)])
            loss.backward()
            opt.step()
        p = np.clip(predict(model, seq[va], pad[va], scalar[va], None if extra is None else extra[va]), 1e-6, 1 - 1e-6)
        val = float(np.mean(-y[va] * np.log(p) - (1 - y[va]) * np.log(1 - p)))
        history.append(val)
        if val < best - 1e-5:
            best, bad = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model = model.cpu()
    model.load_state_dict(best_state)
    return model, history
