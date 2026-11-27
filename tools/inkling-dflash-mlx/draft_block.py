"""Greedy eight-position Inkling DFlash2 draft block for MLX."""
import os

import mlx.core as mx

from inkling_dflash_mxfp8 import linear


WINDOW_LEFT = 2047


def heads(x):
    return x.transpose(1, 0, 2)[None]


def rotate(x, position):
    return mx.fast.rope(heads(x), 128, traditional=False, base=10000,
                        scale=1.0, offset=position)


class DraftBlock:
    """One request's five-layer E4M3 prefix cache, starting at an absolute position."""

    def __init__(self, weights, start=0):
        self.weights = weights
        self.position = start
        self.layers = []

    def norm(self, x, name):
        return mx.fast.rms_norm(x, self.weights[name + '.weight'].astype(x.dtype), 1e-6)

    def project(self, x, name):
        return linear(self.weights, name, x)

    def append(self, target_features):
        """Append committed target rows, shaped [rows, 4 * 2304] in visit order."""
        rows = target_features.shape[0]
        if not rows:
            return
        end = self.position + rows
        features = target_features[-WINDOW_LEFT:]
        start = end - features.shape[0]
        ctx = self.norm(self.project(features, 'fc'), 'hidden_norm')
        updated = []
        for i in range(5):
            name = f'layers.{i}.self_attn.'
            keys = self.norm(self.project(ctx, name + 'k_proj').reshape(-1, 8, 128), name + 'k_norm')
            values = self.project(ctx, name + 'v_proj').reshape(-1, 8, 128)
            pair = mx.to_fp8(rotate(keys, start)), mx.to_fp8(heads(values))
            if self.layers and rows < WINDOW_LEFT:
                pair = tuple(mx.concatenate((old, new), axis=2)[:, :, -WINDOW_LEFT:]
                             for old, new in zip(self.layers[i], pair))
            updated.append(pair)
        self.layers = updated
        self.position = end

    def convolved(self, x, name, operation):
        coefficients = self.project(x, name + '.kernel_projection').reshape(8, 2, 2, 144)
        kernel = self.weights[name + '.base_kernel'].astype(x.dtype)

        def apply(h, stage):
            weights = (coefficients[:, stage, :, :, None] + kernel[stage].reshape(1, 2, 144, 16)).reshape(8, 2, 2304)
            previous = mx.concatenate((mx.zeros_like(h[:1]), h[:-1]))
            return weights[:, 0] * h + weights[:, 1] * previous

        return apply(operation(apply(x, 0)), 1)

    def hidden(self, embeddings):
        """Run the anchor and seven mask embeddings through all five draft layers."""
        if len(self.layers) != 5:
            raise ValueError('Append the committed target prefix before drafting')
        x = embeddings
        for i, (prefix_k, prefix_v) in enumerate(self.layers):
            layer = f'layers.{i}.'
            name = layer + 'self_attn.'

            def attention(h):
                q = self.norm(self.project(h, name + 'q_proj').reshape(8, 32, 128), name + 'q_norm')
                k = self.norm(self.project(h, name + 'k_proj').reshape(8, 8, 128), name + 'k_norm')
                v = self.project(h, name + 'v_proj').reshape(8, 8, 128)
                keys = mx.concatenate((mx.from_fp8(prefix_k, dtype=h.dtype), rotate(k, self.position)), axis=2)
                values = mx.concatenate((mx.from_fp8(prefix_v, dtype=h.dtype), heads(v)), axis=2)
                mask = None
                if self.position + 7 > WINDOW_LEFT:
                    positions = mx.arange(self.position - prefix_k.shape[2], self.position + 8)
                    queries = mx.arange(self.position, self.position + 8)
                    mask = positions[None, :] >= queries[:, None] - WINDOW_LEFT
                out = mx.fast.scaled_dot_product_attention(rotate(q, self.position), keys, values,
                                                           scale=128 ** -0.5, mask=mask)
                return self.project(out[0].transpose(1, 0, 2).reshape(8, -1), name + 'o_proj')

            def mlp(h):
                gate = self.project(h, layer + 'mlp.gate_proj')
                return self.project(gate * mx.sigmoid(gate) * self.project(h, layer + 'mlp.up_proj'),
                                    layer + 'mlp.down_proj')

            x = x + self.convolved(self.norm(x, layer + 'input_layernorm'), layer + 'attention_conv', attention)
            x = x + self.convolved(self.norm(x, layer + 'post_attention_layernorm'), layer + 'mlp_conv', mlp)
        return self.norm(x, 'norm')

    def select(self, hidden, anchor, head):
        """Return seven candidate IDs using the target's packed MXFP8 output head."""
        hidden = hidden[1:]
        scores = mx.quantized_matmul(hidden.astype(mx.bfloat16), *head, mode='mxfp8').astype(mx.float32)
        if os.environ.get('INK_DFLASH_GREEDY_HEAD') == '1':
            return mx.argmax(scores, axis=-1).tolist()
        ids = mx.argpartition(-scores, 15, axis=-1)[:, :16]
        unary = mx.take_along_axis(scores, ids, axis=-1)
        projected = self.project(hidden, 'candidate_selector.hidden_projection')
        previous = mx.concatenate((mx.full((1, 16), anchor), ids[:-1]))
        pred = self.weights['candidate_selector.predecessor_codebook'][previous].astype(projected.dtype)
        succ = self.weights['candidate_selector.successor_codebook'][ids].astype(projected.dtype)
        edges = unary[:, None, :] + (pred * projected[:, None, :]) @ succ.transpose(0, 2, 1)
        path = mx.argmax(30 * mx.tanh(edges / 30), axis=-1).tolist()
        candidates = ids.tolist()
        result = []
        index = 0
        for choices, row in zip(path, candidates):
            index = choices[index]
            result.append(row[index])
        return result
