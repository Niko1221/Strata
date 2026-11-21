// Peer-route status shared by the CUDA/HIP engine and its CPU-only policy test.
#pragma once

namespace strata::core {

enum class PeerRouteStatus {
    forced_host,
    peer_unsupported,
    peer_enable_failed,
    probe_unavailable,
    probe_mismatch,
    peer_verified,
};

constexpr bool peer_route_uses_direct_copy(PeerRouteStatus status) {
    return status == PeerRouteStatus::peer_verified;
}

constexpr const char* peer_route_status_name(PeerRouteStatus status) {
    switch (status) {
        case PeerRouteStatus::forced_host: return "forced_host_fallback";
        case PeerRouteStatus::peer_unsupported: return "peer_access_unsupported";
        case PeerRouteStatus::peer_enable_failed: return "peer_access_enable_failed";
        case PeerRouteStatus::probe_unavailable: return "copy_probe_unavailable";
        case PeerRouteStatus::probe_mismatch: return "copy_probe_data_mismatch";
        case PeerRouteStatus::peer_verified: return "bidirectional_peer_copy_verified";
    }
    return "unknown";
}

}  // namespace strata::core
