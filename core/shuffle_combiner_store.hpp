// Copyright 2016 Husky Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/shuffle_combiner.hpp"
#include "core/zmq_helpers.hpp"

namespace husky {

class ShuffleCombinerSetBase {
   public:
    virtual ~ShuffleCombinerSetBase() {}
};

template <typename KeyT, typename MsgT>
class ShuffleCombinerSet : public ShuffleCombinerSetBase {
   public:
    std::vector<ShuffleCombiner<std::pair<KeyT, MsgT>>> data;
};

class ShuffleCombinerStore {
   public:
    template <typename KeyT, typename MsgT>
    static std::vector<ShuffleCombiner<std::pair<KeyT, MsgT>>>* create_shuffle_combiner(size_t channel_id,
                                                                                        size_t local_id,
                                                                                        size_t num_local_threads,
                                                                                        size_t num_global_threads) {
        // double-checked locking
        if (shuffle_combiners_map.find(channel_id) == shuffle_combiners_map.end()) {
            std::lock_guard<std::mutex> lock(shuffle_combiners_map_mutex);
            if (shuffle_combiners_zmq_context_ptr == nullptr) {
                shuffle_combiners_zmq_context_ptr = new zmq::context_t();
            }
            if (shuffle_combiners_map.find(channel_id) == shuffle_combiners_map.end()) {
                ShuffleCombinerSet<KeyT, MsgT>* shuffle_combiner_set = new ShuffleCombinerSet<KeyT, MsgT>();
                shuffle_combiner_set->data.resize(num_local_threads);
                ShuffleCombiner<std::pair<KeyT, MsgT>>::init_sockets(num_local_threads, channel_id,
                                                                     *shuffle_combiners_zmq_context_ptr);
                for (int i = 0; i < num_local_threads; i++) {
                    shuffle_combiner_set->data[i].init(num_global_threads, num_local_threads, channel_id, i);
                }
                ShuffleCombinerStore::num_local_threads.insert(std::make_pair(channel_id, num_local_threads));
                shuffle_combiners_map.insert(std::make_pair(channel_id, shuffle_combiner_set));
            }
        }
        auto& data = dynamic_cast<ShuffleCombinerSet<KeyT, MsgT>*>(shuffle_combiners_map[channel_id])->data;
        return &data;
    }

    template <typename KeyT, typename MsgT>
    static void remove_shuffle_combiner(size_t channel_id) {
        std::lock_guard<std::mutex> lock(shuffle_combiners_map_mutex);
        num_local_threads[channel_id] -= 1;
        if (num_local_threads[channel_id] == 0) {
            ShuffleCombiner<std::pair<KeyT, MsgT>>::erase_key(channel_id);
            delete shuffle_combiners_map[channel_id];
            shuffle_combiners_map.erase(channel_id);
            num_local_threads.erase(channel_id);
        }
        if (shuffle_combiners_map.empty()) {
            delete shuffle_combiners_zmq_context_ptr;
            shuffle_combiners_zmq_context_ptr = nullptr;
        }
    }

   protected:
    static std::unordered_map<size_t, ShuffleCombinerSetBase*> shuffle_combiners_map;
    static std::mutex shuffle_combiners_map_mutex;
    static std::unordered_map<size_t, size_t> num_local_threads;
    static zmq::context_t* shuffle_combiners_zmq_context_ptr;
};

}  // namespace husky
