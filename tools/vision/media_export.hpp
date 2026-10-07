#pragma once

#include "strata/program/media_embeddings.hpp"
#include <cstdint>
#include <string>

struct mtmd_context;
struct llama_model;

namespace strata::vision {

inline constexpr const char* video_profile_name = "qwen4_exp_16x2x2_2560_v1";
struct VideoExportLimits {
    uint32_t group_tokens = 256;
    uint32_t frames = 128;
    uint64_t rows = 16384;
    uint64_t rgb_bytes = 256u << 20;
    uint64_t embedding_bytes = 256u << 20;
    double duration_s = 60.0;
};

bool supports_video_profile(const std::string& projector, const llama_model* model, int width);
program::MediaBundle export_video(mtmd_context* ctx, const std::string& packet, uint32_t width,
                                  const VideoExportLimits& limits);
void publish_media(const std::string& output, const program::MediaBundle& bundle);
void warm_video(mtmd_context* ctx, uint32_t width, const VideoExportLimits& limits);

}  // namespace strata::vision
