// GLM-5.3-Flash's vision tower (transformers' Glm5NextVisionModel), added to llama.cpp's mtmd by
// tools/vision/glm5next_patch.py.  GLM-4V's graph (models/glm4v.cpp) without the norm after the patch embedding and
// without learned position embeddings; the per-head q/k RMSNorm comes in through build_vit (the attn_q_norm /
// attn_k_norm tensors), and the SwiGLU clamp of the blocks and the merger through hparams.ffn_clamp in build_ffn.
//   patches (2 frames of 14x14, as two conv2d) -> 24 blocks (RMSNorm, fused qkv, q/k norm per head, axial 2D rope =
//   Qwen2-VL's vision rope, SwiGLU clamped at 10) -> RMSNorm -> 2x2 conv to 4096 -> linear, LayerNorm, GELU (erf),
//   SwiGLU 4096 -> 10240 -> 4096 (clamped)
#include "models.h"

ggml_cgraph * clip_graph_glm5next::build() {
    GGML_ASSERT(model.patch_bias != nullptr);
    GGML_ASSERT(model.patch_embeddings_1 != nullptr);

    const int batch_size = 1;
    const norm_type norm_t = NORM_TYPE_RMS;

    ggml_tensor * inp_raw = build_inp_raw();
    ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);

    int mrope_sections[4] = {d_head/4, d_head/4, d_head/4, d_head/4};
    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches * 4);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    GGML_ASSERT(img.nx() % (patch_size * 2) == 0);
    GGML_ASSERT(img.ny() % (patch_size * 2) == 0);

    // the second frame, then the patches in the processor's order: 2x2 merge windows, row-major inside each
    {
        auto inp_1 = ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
        inp = ggml_add(ctx0, inp, inp_1);

        inp = ggml_permute(ctx0, inp, 1, 2, 0, 3);  // [w, h, c, b] -> [c, w, h, b]
        inp = ggml_cont_4d(ctx0, inp, n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
        inp = ggml_reshape_4d(ctx0, inp, n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
        inp = ggml_permute(ctx0, inp, 0, 2, 1, 3);
        inp = ggml_cont_3d(ctx0, inp, n_embd, n_patches_x * n_patches_y, batch_size);
    }

    inp = ggml_add(ctx0, inp, model.patch_bias);
    cb(inp, "patch_bias", -1);

    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        return ggml_rope_multi(
                    ctx0, cur, positions, nullptr,
                    d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION,
                    32768, hparams.rope_theta, 1, 0, 1, 32, 1);
    };

    ggml_tensor * cur = build_vit(inp, n_patches, norm_t, hparams.ffn_op, nullptr, add_pos);
    cb(cur, "vit_out", -1);

    // the 2x2 downsample conv
    {
        const int n_merge = hparams.n_merge;
        GGML_ASSERT(n_merge > 0);
        const int n_token_out = n_patches / n_merge / n_merge;
        cur = ggml_reshape_4d(ctx0, cur, n_embd, n_merge, n_merge, n_token_out);
        cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 2, 0, 1, 3));   // [n_merge, n_merge, n_embd, n_token_out]
        cur = ggml_conv_2d(ctx0, model.mm_patch_merger_w, cur, n_merge, n_merge, 0, 0, 1, 1);
        cur = ggml_reshape_2d(ctx0, cur, cur->ne[2], n_token_out);   // [n_embd_out, n_token_out]
        cur = ggml_add(ctx0, cur, model.mm_patch_merger_b);
    }

    // the merger: linear, LayerNorm, GELU, then the clamped SwiGLU
    {
        cur = build_mm(model.mm_fc_w, cur);
        cur = build_norm(cur, model.mm_post_norm_w, model.mm_post_norm_b, NORM_TYPE_NORMAL, 1e-5, -1);
        cur = ggml_gelu_erf(ctx0, cur);
        cb(cur, "after_fc_proj", -1);
        cur = build_ffn(cur,
            model.mm_ffn_up_w, model.mm_ffn_up_b,
            model.mm_ffn_gate_w, model.mm_ffn_gate_b,
            model.mm_ffn_down_w, model.mm_ffn_down_b,
            hparams.ffn_op, -1);
        cb(cur, "after_ffn_proj", -1);
    }

    ggml_build_forward_expand(gf, cur);
    return gf;
}
