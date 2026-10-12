#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace strata::core {

// Snapshot-owned host storage. Copies share immutable segments; a mutable visit
// detaches the touched segment before exposing a writable pointer. Appending
// never relocates existing payloads and only the small directory can move.
class ConversationBuffer {
public:
    static constexpr size_t segment_bytes = 16 * 1024 * 1024;
    using Payload = std::vector<uint8_t>;
    struct Segment {
        std::shared_ptr<Payload> payload;
        size_t offset = 0, length = 0;
        bool published = false;
    };
    ConversationBuffer() = default;
    ConversationBuffer(std::initializer_list<uint8_t> values) {
        resize(values.size());
        visit(0, size(), [&](uint8_t* p, size_t n, size_t at) {
            std::memcpy(p, values.begin() + at, n);
            return true;
        });
    }
    ConversationBuffer(const ConversationBuffer&) = default;
    ConversationBuffer& operator=(const ConversationBuffer&) = default;
    ConversationBuffer(ConversationBuffer&& other) noexcept
        : segments_(std::move(other.segments_)), size_(std::exchange(other.size_, 0)) {}
    ConversationBuffer& operator=(ConversationBuffer&& other) noexcept {
        if (this == &other) return *this;
        segments_ = std::move(other.segments_);
        size_ = std::exchange(other.size_, 0);
        return *this;
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    void freeze() { for (auto& segment : segments_) segment.published = true; }
    size_t directory_bytes() const { return segments_.capacity() * sizeof(Segment); }
    template<class Fn> void for_each_payload(Fn fn) const {
        for (size_t i = 0; i < segments_.size(); ++i) {
            bool seen = false;
            for (size_t j = 0; j < i; ++j)
                if (segments_[j].payload == segments_[i].payload) { seen = true; break; }
            if (!seen) fn(segments_[i].payload.get(), segments_[i].payload->capacity(),
                          (size_t)segments_[i].payload.use_count());
        }
    }
    size_t bytes() const {
        size_t n = directory_bytes();
        for_each_payload([&](const void*, size_t capacity, size_t) { n += capacity; });
        return n;
    }
    size_t shared_bytes() const {
        size_t n = 0;
        for (size_t i = 0; i < segments_.size(); ++i) {
            size_t local = 0;
            bool seen = false;
            for (size_t j = 0; j < segments_.size(); ++j) {
                if (segments_[j].payload == segments_[i].payload) {
                    ++local;
                    if (j < i) seen = true;
                }
            }
            if (!seen && segments_[i].payload.use_count() > (long)local)
                n += segments_[i].payload->capacity();
        }
        return n;
    }
    static size_t copied_bytes() { return copied_bytes_.load(std::memory_order_relaxed); }
    // Includes the old directory while reserve allocates its replacement.
    // Saturation makes an overflowing estimate fail admission before resize.
    size_t allocation_peak(size_t n) const {
        size_t total = bytes();
        // Capture rewrites may detach published or shared views after resize.
        // Admission includes those payloads and replacement view directories,
        // even for a same-size capture or rewind with no growth allocation.
        size_t retained = std::min(n, size_), detachable = 0;
        for (const auto& segment : segments_) {
            const size_t length = std::min(retained, segment.length);
            if (length && (segment.published || segment.payload.use_count() > 1)) {
                if (!add(total, length) || !add(detachable, 1)) return SIZE_MAX;
            }
            retained -= length;
        }
        if (detachable) {
            size_t views = segments_.size();
            if (detachable > SIZE_MAX / 2 || !add(views, detachable * 2) ||
                views > SIZE_MAX / sizeof(Segment) ||
                !add(total, views * sizeof(Segment))) return SIZE_MAX;
        }
        if (n <= size_) return total;
        const size_t spare = writable_back() ? segments_.back().payload->capacity() -
                                                   segments_.back().payload->size() : 0;
        const size_t extra = n - size_ - std::min(n - size_, spare);
        const size_t full = extra / segment_bytes, tail = extra % segment_bytes;
        if (!add(total, full * segment_bytes)) return SIZE_MAX;
        if (tail && !add(total, segment_capacity(tail, full ? segment_bytes :
                segments_.empty() ? 0 : segments_.back().payload->capacity(), size_ != 0))) return SIZE_MAX;
        size_t count = segments_.size();
        if (!add(count, full + (tail != 0))) return SIZE_MAX;
        if (count > segments_.capacity() &&
            (count > SIZE_MAX / sizeof(Segment) || !add(total, count * sizeof(Segment)))) return SIZE_MAX;
        return total;
    }
    void resize(size_t n, uint8_t value = 0) {
        if (n > size_) {
            const bool growing = size_ != 0;
            const size_t spare = writable_back() ? segments_.back().payload->capacity() -
                                                       segments_.back().payload->size() : 0;
            const size_t extend = std::min(n - size_, spare);
            const size_t extra = n - size_ - extend;
            segments_.reserve(segments_.size() + extra / segment_bytes + (extra % segment_bytes != 0));
            if (extend) {
                auto& segment = segments_.back();
                segment.payload->resize(segment.payload->size() + extend, value);
                segment.length += extend;
                size_ += extend;
            }
            while (size_ < n) {
                const size_t count = std::min(segment_bytes, n - size_);
                const size_t previous = segments_.empty() ? 0 : segments_.back().payload->capacity();
                auto payload = std::make_shared<Payload>();
                payload->reserve(segment_capacity(count, previous, growing));
                payload->resize(count, value);
                segments_.push_back(Segment{std::move(payload), 0, count, false});
                size_ += count;
            }
        } else {
            while (!segments_.empty() && size_ - segments_.back().length >= n) {
                size_ -= segments_.back().length;
                segments_.pop_back();
            }
            if (size_ > n) segments_.back().length -= size_ - n;
            size_ = n;
        }
    }
    void pop_back() { resize(size_ - 1); }

    // fn(pointer, count, absolute_offset); false aborts a transfer immediately.
    template<class Fn> bool visit(size_t offset, size_t count, Fn fn) {
        if constexpr (std::is_invocable_r_v<bool, Fn, const uint8_t*, size_t, size_t>)
            return visit_impl(static_cast<const ConversationBuffer&>(*this), offset, count, fn);
        else
            return visit_impl(*this, offset, count, fn);
    }
    template<class Fn> bool visit(size_t offset, size_t count, Fn fn) const {
        return visit_impl(*this, offset, count, fn);
    }
    bool read(void* target, size_t offset, size_t count) const {
        return visit(offset, count, [&](const uint8_t* p, size_t n, size_t at) {
            std::memcpy(static_cast<uint8_t*>(target) + at - offset, p, n);
            return true;
        });
    }
    bool operator==(const ConversationBuffer& other) const {
        if (size_ != other.size_) return false;
        size_t a = 0, b = 0, x = 0, y = 0;
        while (a < segments_.size() && b < other.segments_.size()) {
            const auto& left = segments_[a];
            const auto& right = other.segments_[b];
            const size_t n = std::min(left.length - x, right.length - y);
            if (std::memcmp(left.payload->data() + left.offset + x,
                            right.payload->data() + right.offset + y, n)) return false;
            x += n; y += n;
            if (x == left.length) { ++a; x = 0; }
            if (y == right.length) { ++b; y = 0; }
        }
        return true;
    }

private:
    // A fresh capture allocates exactly its payload. Later appends reserve
    // geometrically growing segments, bounded by 16 MiB. Small chat turns then
    // extend the final allocation instead of adding one transfer per turn.
    static size_t segment_capacity(size_t count, size_t previous, bool growing) {
        if (!growing) return count;
        return std::max(count, std::min(segment_bytes, std::max<size_t>(65536, std::min(previous, segment_bytes/2)*2)));
    }
    static bool add(size_t& n, size_t extra) {
        if (extra > SIZE_MAX - n) return false;
        n += extra;
        return true;
    }
    size_t local_refs(const Payload* payload) const {
        size_t n = 0;
        for (const auto& segment : segments_) n += segment.payload.get() == payload;
        return n;
    }
    bool writable_back() const {
        if (segments_.empty()) return false;
        const auto& segment = segments_.back();
        return !segment.published && segment.offset == 0 &&
               segment.length == segment.payload->size() &&
               segment.payload.use_count() == (long)local_refs(segment.payload.get());
    }
    void prepare_write(size_t offset, size_t count) {
        size_t begin = 0;
        for (size_t i = 0; i < segments_.size() && count;) {
            const size_t length = segments_[i].length;
            const size_t end = begin + length;
            if (offset < end) {
                const size_t local = offset - begin;
                const size_t n = std::min(count, end - offset);
                auto& segment = segments_[i];
                const size_t owned_refs = local_refs(segment.payload.get());
                const bool shared = segment.payload.use_count() > (long)owned_refs;
                if (shared && !segment.published) {
                    // Working value copies remain contiguous until publication.
                    // Frozen checkpoints use bounded written-range copies below.
                    auto private_payload = std::make_shared<Payload>(
                        segment.payload->begin() + segment.offset,
                        segment.payload->begin() + segment.offset + length);
                    copied_bytes_.fetch_add(length, std::memory_order_relaxed);
                    segment = Segment{std::move(private_payload), 0, length, false};
                    ++i;
                } else if (shared || segment.published) {
                    // Keep values, not a reference into the directory: reserve
                    // can relocate its Segment objects.
                    const auto payload = segment.payload;
                    const size_t payload_offset = segment.offset;
                    const bool published = segment.published;
                    // Reserve before replacing the old view so allocation
                    // failure leaves the buffer's published layout intact.
                    segments_.reserve(segments_.size() + 2);
                    std::vector<Segment> parts;
                    parts.reserve(3);
                    if (local) parts.push_back(Segment{payload, payload_offset, local, published});
                    auto private_payload = std::make_shared<Payload>(
                        payload->begin() + payload_offset + local,
                        payload->begin() + payload_offset + local + n);
                    copied_bytes_.fetch_add(n, std::memory_order_relaxed);
                    parts.push_back(Segment{std::move(private_payload), 0, n, false});
                    if (local + n < length)
                        parts.push_back(Segment{payload, payload_offset + local + n,
                                                length - local - n, published});
                    segments_.erase(segments_.begin() + (std::ptrdiff_t)i);
                    segments_.insert(segments_.begin() + (std::ptrdiff_t)i, parts.begin(), parts.end());
                    i += parts.size();
                } else {
                    ++i;
                }
                offset += n;
                count -= n;
            } else {
                ++i;
            }
            begin = end;
        }
    }
    template<class Self, class Fn> static bool visit_impl(Self& self, size_t offset, size_t count, Fn fn) {
        if (offset > self.size_ || count > self.size_ - offset) return false;
        if constexpr (!std::is_const_v<Self>) self.prepare_write(offset, count);
        size_t begin = 0;
        for (size_t i = 0; i < self.segments_.size(); ++i) {
            if (!count) break;
            const auto& segment = self.segments_[i];
            const size_t end = begin + segment.length;
            if (offset < end) {
                const size_t n = std::min(count, end - offset);
                if constexpr (!std::is_const_v<Self>) {
                    auto& writable = self.segments_[i];
                    if (!fn(writable.payload->data() + writable.offset + offset - begin, n, offset)) return false;
                } else {
                    const uint8_t* data = segment.payload->data() + segment.offset + offset - begin;
                    if (!fn(data, n, offset)) return false;
                }
                offset += n; count -= n;
            }
            begin = end;
        }
        return count == 0;
    }
    inline static std::atomic<size_t> copied_bytes_{0};
    std::vector<Segment> segments_;
    size_t size_ = 0;
};

} // namespace strata::core
