"""Compare chunked and contiguous draft prefixes on deterministic synthetic features."""
import argparse

import mlx.core as mx

from draft_block import DraftBlock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('packed_checkpoint')
    args = parser.parse_args()
    mx.random.seed(17)
    weights = mx.load(args.packed_checkpoint)
    features = mx.random.normal((40, 9216))
    embeddings = mx.random.normal((8, 2304))
    head = mx.quantize(mx.random.normal((1024, 2304)).astype(mx.bfloat16), mode='mxfp8')
    paths = []
    for pieces in ((features,), (features[:24], features[24:])):
        block = DraftBlock(weights, start=2030)
        for piece in pieces:
            block.append(piece)
        paths.append(block.select(block.hidden(embeddings), 23, head))
    if paths[0] != paths[1]:
        raise ValueError(f'Candidate disagreement: {paths}')
    print(paths[0])


if __name__ == '__main__':
    main()
