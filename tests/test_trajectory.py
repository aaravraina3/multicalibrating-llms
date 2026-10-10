import numpy as np
import torch

from calib.trajectory import N_CHUNKS, TrajectoryNet, chunk_sequence


def test_chunking_long_and_short():
    lp = np.linspace(-3, 0, 1000).astype(np.float32)
    seq, pad = chunk_sequence(lp, 500, 1000)
    assert seq.shape == (N_CHUNKS, 4) and not pad.any()
    assert seq[0, 3] == 0 and seq[-1, 3] == 1  # code span share
    seq, pad = chunk_sequence(lp[:10], 0, 0)
    assert pad.sum() == N_CHUNKS - 10 and np.all(seq[pad] == 0)


def test_padding_does_not_change_output():
    torch.manual_seed(0)
    net = TrajectoryNet().eval()
    seq = torch.randn(1, N_CHUNKS, 4)
    pad = torch.zeros(1, N_CHUNKS, dtype=torch.bool)
    pad[0, 20:] = True
    scalar = torch.zeros(1)
    a = net(seq, pad, scalar)
    seq2 = seq.clone()
    seq2[0, 20:] = 99.0  # garbage in padded positions
    assert torch.allclose(a, net(seq2, pad, scalar), atol=1e-5)
