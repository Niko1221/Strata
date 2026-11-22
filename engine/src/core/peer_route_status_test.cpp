#include "strata/core/peer_route_status.hpp"

#include <array>
#include <cstdio>
#include <cstring>

using strata::core::PeerRouteStatus;
using strata::core::peer_route_status_name;
using strata::core::peer_route_uses_direct_copy;

int main() {
    if (!peer_route_uses_direct_copy(PeerRouteStatus::peer_verified)) return 1;
    for (const PeerRouteStatus status : std::array{
             PeerRouteStatus::forced_host,
             PeerRouteStatus::peer_unsupported,
             PeerRouteStatus::peer_enable_failed,
             PeerRouteStatus::probe_unavailable,
             PeerRouteStatus::probe_mismatch,
         }) {
        if (peer_route_uses_direct_copy(status)) return 2;
        if (std::strcmp(peer_route_status_name(status), "unknown") == 0) return 3;
    }
    std::puts("peer_route_status_test: ok");
    return 0;
}
