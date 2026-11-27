"""Pack and execute Inkling-2-Mini DFlash2 linear projections with MLX."""
import argparse
from pathlib import Path

import mlx.core as mx


def pack_checkpoint(source, output):
    if output.exists():
        raise FileExistsError(output)
    checkpoint = mx.load(str(source))
    if 'fc.weight' not in checkpoint or 'candidate_selector.hidden_projection.weight' not in checkpoint:
        raise ValueError('Expected an Inkling-2-Mini DFlash2 draft checkpoint')
    packed = {}
    for name, tensor in checkpoint.items():
        is_projection = (name.endswith('.weight') and tensor.ndim == 2
                         and tensor.shape[1] % 32 == 0 and 'codebook' not in name)
        if is_projection:
            weight, scales = mx.quantize(tensor.astype(mx.bfloat16), mode='mxfp8')
            mx.eval(weight, scales)
            packed[name] = weight
            packed[name + '.scales'] = scales
        else:
            packed[name] = tensor
    mx.save_safetensors(str(output), packed)


def linear(checkpoint, name, inputs):
    """Apply one named projection, returning activations in the input dtype."""
    weight = checkpoint[name + '.weight']
    scales = checkpoint.get(name + '.weight.scales')
    if scales is None:
        return inputs @ weight.astype(inputs.dtype).T
    result = mx.quantized_matmul(inputs.astype(mx.bfloat16), weight, scales, mode='mxfp8')
    return result.astype(inputs.dtype)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    pack_checkpoint(args.source, args.output)
