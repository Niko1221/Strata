#include "strata/core/expert_profile_memory.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>

namespace strata::core {
namespace {

bool read_exact(std::FILE* f, void* data, size_t size) {
    return size == 0 || std::fread(data, 1, size, f) == size;
}

bool write_exact(std::FILE* f, const void* data, size_t size) {
    return size == 0 || std::fwrite(data, 1, size, f) == size;
}

bool valid_heat(const std::vector<double>& heat) {
    return std::all_of(heat.begin(), heat.end(), [](double v) { return std::isfinite(v) && v >= 0.0; });
}

}  // namespace

void ExpertProfileMemory::reset(int64_t n_layers, int64_t n_expert, double now) {
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    updated_at_ = now;
    const size_t n = (size_t) (n_layers * n_expert);
    cell_.assign(n, 0.0);
    committed_heat_.assign(n, 0.0);
    scores_.assign(n, 0.0);
}

bool ExpertProfileMemory::load(const std::string& path, int64_t n_layers, int64_t n_expert, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "expert profile memory: cannot open " + path;
        return false;
    }
    char magic[4] = {};
    uint32_t header[3] = {};
    double updated_at = 0.0;
    const size_t n = (size_t) (n_layers * n_expert);
    std::vector<double> cell(n);
    bool ok = read_exact(f, magic, 4) && read_exact(f, header, sizeof header) &&
              read_exact(f, &updated_at, sizeof updated_at) && read_exact(f, cell.data(), n * sizeof(double));
    char extra;
    ok = ok && std::fread(&extra, 1, 1, f) == 0;
    std::fclose(f);
    if (!ok || std::memcmp(magic, "STRM", 4) != 0 || header[0] != 1 ||
        (int64_t) header[1] != n_layers || (int64_t) header[2] != n_expert || !std::isfinite(updated_at) ||
        updated_at < 0.0 || !valid_heat(cell)) {
        err = "expert profile memory: " + path + " is truncated, corrupt, or belongs to another model";
        return false;
    }
    reset(n_layers, n_expert, updated_at);
    cell_ = std::move(cell);
    scores_ = cell_;
    return true;
}

bool ExpertProfileMemory::save(const std::string& path, std::string& err) const {
    if (n_layers_ <= 0 || n_expert_ <= 0 || cell_.size() != (size_t) (n_layers_ * n_expert_)) {
        err = "expert profile memory: no initialized state to save";
        return false;
    }
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        err = "expert profile memory: cannot create " + tmp;
        return false;
    }
    const uint32_t header[3] = {1, (uint32_t) n_layers_, (uint32_t) n_expert_};
    bool ok = write_exact(f, "STRM", 4) && write_exact(f, header, sizeof header) &&
              write_exact(f, &updated_at_, sizeof updated_at_) &&
              write_exact(f, cell_.data(), cell_.size() * sizeof(double));
    ok = (std::fclose(f) == 0) && ok;
    std::error_code ec;
    if (ok) std::filesystem::rename(tmp, path, ec);
    if (!ok || ec) {
        std::filesystem::remove(tmp, ec);
        err = "expert profile memory: cannot write " + path;
        return false;
    }
    return true;
}

bool ExpertProfileMemory::consolidate(const std::vector<double>& heat, double now, double half_life_hours,
                                      double long_term_weight, std::string& err) {
    const size_t n = (size_t) (n_layers_ * n_expert_);
    if (n_layers_ <= 0 || n_expert_ <= 0 || heat.size() != n || cell_.size() != n || committed_heat_.size() != n ||
        !valid_heat(heat) || !std::isfinite(now) || now < 0.0 || !(half_life_hours > 0.0) ||
        !std::isfinite(half_life_hours) || !(long_term_weight >= 0.0 && long_term_weight <= 1.0) ||
        !std::isfinite(long_term_weight)) {
        err = "expert profile memory: invalid heat, time, half-life, or output weight";
        return false;
    }
    std::vector<double> delta(n);
    double total = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (heat[i] < committed_heat_[i]) {
            err = "expert profile memory: cumulative heat went backwards";
            return false;
        }
        delta[i] = heat[i] - committed_heat_[i];
        total += delta[i];
    }
    if (!(total > 0.0)) return true;
    if (!std::isfinite(total)) {
        err = "expert profile memory: total heat is not finite";
        return false;
    }

    // An empty cell accepts the first observation in full. Later updates use elapsed wall time, so a five-minute
    // autosave is not 288 times as much evidence as one save a day. No routing activity means no consolidation.
    const double elapsed = std::max(0.0, now - updated_at_);
    const bool empty = std::all_of(cell_.begin(), cell_.end(), [](double v) { return v == 0.0; });
    const double forget = empty ? 0.0 : std::exp2(-elapsed / (half_life_hours * 3600.0));
    const double input = 1.0 - forget;
    for (size_t i = 0; i < n; ++i) {
        const double recent = delta[i] / total;
        cell_[i] = forget * cell_[i] + input * recent;
        scores_[i] = long_term_weight * cell_[i] + (1.0 - long_term_weight) * recent;
    }
    committed_heat_ = heat;
    updated_at_ = now;
    return true;
}

}  // namespace strata::core
