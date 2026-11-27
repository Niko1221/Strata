"""Serve verified-prefix draft requests from a local Strata Metal process."""
import socket
import struct
import sys

import mlx.core as mx
import numpy as np

from draft_block import DraftBlock
from sq10 import base, embed


def serve(sock):
    mx.set_cache_limit(32 * 1024 * 1024)
    weights = mx.load(str(base / 'draft-mxfp8.safetensors'))
    packed = mx.load(str(base / 'draft-head-mxfp8.safetensors'))
    head = packed['weight'], packed['scales']
    header = struct.Struct('=6i')
    slots = {}
    print('READY', flush=True)
    with sock.makefile('rb') as stream:
        while message := stream.read(header.size):
            start, rows, anchor, count, slot, reset = header.unpack(message)
            if not (0 < rows <= 2047 and 0 < count <= 7):
                raise ValueError(f'Invalid draft dimensions {rows}, {count}')
            if reset:
                slots[slot] = DraftBlock(weights, start)
            block = slots[slot]
            if block.position != start:
                raise ValueError(f'Expected position {block.position}, received {start}')
            data = stream.read(rows * 4 * 2304 * 4)
            features = np.frombuffer(data, '<f4').reshape(4, rows, 2304).transpose(1, 0, 2)
            block.append(mx.array(features.reshape(rows, 9216)))
            hidden = block.hidden(embed([anchor] + [200056] * 7))
            candidates = block.select(hidden, anchor, head)
            sock.sendall(struct.pack(f'={count}i', *candidates[:count]))


if __name__ == '__main__':
    with socket.socket(fileno=int(sys.argv[1])) as channel:
        serve(channel)
