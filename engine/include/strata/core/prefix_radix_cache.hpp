// MIT License - Copyright (c) 2026 Strata contributors
#pragma once

#include <algorithm>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <vector>

namespace strata::core {

// Single scheduler thread. Only materialized boundaries can be restored.
template <class Payload>
class PrefixRadixCache {
    struct Node;
    using Lru = std::list<Node*>;
    struct Node {
        Node* parent = nullptr;
        std::vector<int32_t> edge;
        std::map<int32_t, std::unique_ptr<Node>> children;
        std::unique_ptr<Payload> state;
        typename Lru::iterator recent;
        size_t charge = 0;
    };

public:
    explicit PrefixRadixCache(size_t limit) : limit_(limit) {}

    template <class Token, class Compatible>
    const Payload* find(const std::vector<Token>& tokens, size_t length, Compatible compatible, bool touch) {
        Node* node = &root_;
        Node* best = nullptr;
        size_t pos = 0;
        while (pos < length) {
            node = match_edge(node, tokens, length, pos);
            if (!node) break;
            if (node->state && compatible(*node->state)) best = node;
        }
        if (!best) return nullptr;
        if (touch) recent_.splice(recent_.begin(), recent_, best->recent);
        return best->state.get();
    }

    Payload* find_exact(const std::vector<int32_t>& tokens) {
        Node* node = exact_node(tokens);
        return node ? node->state.get() : nullptr;
    }

    void erase_exact(const std::vector<int32_t>& tokens) {
        Node* node = exact_node(tokens);
        if (node && node->state) remove(node);
    }

    // Allow for edge-vector capacity and two tree/map nodes (a leaf and a split).
    static size_t index_bytes(size_t length) {
        return 2 * length * sizeof(int32_t) + 2 * (sizeof(Node) + 8 * sizeof(void*));
    }

    template <class SharedBytes>
    bool reserve(size_t additional, size_t pinned, SharedBytes shared) {
        if (additional > limit_ || pinned > limit_ - additional) return false;
        while (private_bytes_ + shared() > limit_ - additional) {
            if (recent_.empty()) return false;
            evict();
        }
        return true;
    }

    void insert(const std::vector<int32_t>& tokens, std::unique_ptr<Payload> state) {
        const size_t length = state->length;
        Node* node = &root_;
        size_t pos = 0;
        while (pos < length) {
            auto it = node->children.find(tokens[pos]);
            if (it == node->children.end()) {
                auto leaf = std::make_unique<Node>();
                leaf->parent = node;
                leaf->edge.assign(tokens.begin() + pos, tokens.begin() + length);
                node = node->children.emplace(tokens[pos], std::move(leaf)).first->second.get();
                break;
            }
            auto& child = it->second;
            const size_t n = std::min(child->edge.size(), length - pos);
            size_t common = 0;
            while (common < n && child->edge[common] == tokens[pos + common]) ++common;
            if (common < child->edge.size()) {
                auto split = std::make_unique<Node>();
                split->parent = node;
                split->edge.assign(child->edge.begin(), child->edge.begin() + common);
                auto& suffix = split->children[child->edge[common]];
                child->edge.erase(child->edge.begin(), child->edge.begin() + common);
                child->parent = split.get();
                suffix = std::move(child);
                child = std::move(split);
            }
            node = child.get();
            pos += common;
        }
        if (node->state) {
            private_bytes_ -= node->charge;
            recent_.erase(node->recent);
        }
        recent_.push_front(node);
        node->recent = recent_.begin();
        node->charge = state->bytes() + index_bytes(length);
        private_bytes_ += node->charge;
        node->state = std::move(state);
    }

    void clear() {
        recent_.clear();
        root_.children.clear();
        private_bytes_ = 0;
    }

private:
    template <class Token>
    static Node* match_edge(Node* parent, const std::vector<Token>& tokens, size_t length, size_t& pos) {
        const auto child = parent->children.find(static_cast<int32_t>(tokens[pos]));
        if (child == parent->children.end()) return nullptr;
        Node* node = child->second.get();
        if (node->edge.size() > length - pos ||
            !std::equal(node->edge.begin(), node->edge.end(), tokens.begin() + pos)) return nullptr;
        pos += node->edge.size();
        return node;
    }

    Node* exact_node(const std::vector<int32_t>& tokens) {
        Node* node = &root_;
        size_t pos = 0;
        while (node && pos < tokens.size()) node = match_edge(node, tokens, tokens.size(), pos);
        return node;
    }

    void evict() { remove(recent_.back()); }

    void remove(Node* node) {
        recent_.erase(node->recent);
        private_bytes_ -= node->charge;
        node->state.reset();
        while (node != &root_ && !node->state) {
            auto* parent = node->parent;
            const int32_t key = node->edge.front();
            if (node->children.size() > 1) break;
            if (node->children.empty()) {
                parent->children.erase(key);
                node = parent;
                continue;
            }
            // Preserve the child's address: the LRU may still point to it.
            auto& child = node->children.begin()->second;
            node->edge.insert(node->edge.end(), child->edge.begin(), child->edge.end());
            child->edge.swap(node->edge);
            child->parent = parent;
            auto merged = std::move(child);
            parent->children[key] = std::move(merged);
            break;
        }
    }

    size_t limit_, private_bytes_ = 0;
    Node root_;
    Lru recent_;
};

} // namespace strata::core
