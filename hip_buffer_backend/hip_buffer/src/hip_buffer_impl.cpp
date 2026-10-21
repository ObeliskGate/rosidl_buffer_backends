// Copyright 2026 Open Source Robotics Foundation, Inc.
// Modified for the HIP port by ObeliskGate, 2026.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "hip_buffer/hip_buffer_impl.hpp"

#include <memory>

namespace hip_buffer_backend
{

std::shared_ptr<HipMemoryPool> get_or_create_global_pool()
{
  static std::shared_ptr<HipMemoryPool> global_pool = [] {
      auto pool = std::make_shared<HipMemoryPool>();
      const hipError_t result = pool->create();
      if (result != hipSuccess) {
        throw HipError(__FILE__, __LINE__, "HipMemoryPool::create", result);
      }
      return pool;
    }();
  return global_pool;
}

}  // namespace hip_buffer_backend
