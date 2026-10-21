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

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <cstring>
#include <vector>

#include "hip_buffer/hip_buffer_api.hpp"
#include "rosidl_buffer/buffer.hpp"

class HipBufferTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    hipStreamCreate(&stream1_);
    hipStreamCreate(&stream2_);
  }

  void TearDown() override
  {
    hipStreamDestroy(stream1_);
    hipStreamDestroy(stream2_);
  }

  void allocate_buffer(rosidl::Buffer<uint8_t> & buffer, size_t count)
  {
    auto impl = std::make_unique<hip_buffer_backend::HipBufferImpl<uint8_t>>(count);
    buffer = rosidl::Buffer<uint8_t>(std::move(impl));
  }

  void write_pattern(uint8_t * device_ptr, size_t count, uint8_t offset, hipStream_t stream)
  {
    std::vector<uint8_t> host(count);
    for (size_t i = 0; i < count; ++i) {
      host[i] = static_cast<uint8_t>((offset + i) % 256);
    }
    hipMemcpyAsync(device_ptr, host.data(), count, hipMemcpyHostToDevice, stream);
  }

  std::vector<uint8_t> read_to_host(const uint8_t * device_ptr, size_t count, hipStream_t stream)
  {
    std::vector<uint8_t> host(count);
    hipMemcpyAsync(host.data(), device_ptr, count, hipMemcpyDeviceToHost, stream);
    hipStreamSynchronize(stream);
    return host;
  }

  hipStream_t stream1_{nullptr};
  hipStream_t stream2_{nullptr};
};

TEST_F(HipBufferTest, AllocateAndWriteHandle)
{
  rosidl::Buffer<uint8_t> buffer;
  allocate_buffer(buffer, 1024);

  hip_buffer_backend::WriteHandle handle =
    hip_buffer_backend::from_output_buffer(buffer, stream1_);

  EXPECT_NE(nullptr, handle.get_ptr());
  EXPECT_EQ("hip", buffer.get_backend_type());
  EXPECT_EQ(1024u, buffer.size());
}

TEST_F(HipBufferTest, FromBuffer_PromotesCpuBufferForRead)
{
  rosidl::Buffer<uint8_t> buffer(64);
  for (size_t i = 0; i < 64; ++i) {
    buffer[i] = static_cast<uint8_t>(0xA0 + (i % 16));
  }
  const rosidl::Buffer<uint8_t> & cbuf = buffer;

  hip_buffer_backend::ReadHandle rh =
    hip_buffer_backend::from_input_buffer(cbuf, stream1_);
  auto promoted = rh.get_promoted_buffer();
  ASSERT_NE(nullptr, promoted);
  EXPECT_EQ(promoted->get_backend_type(), "hip");
  EXPECT_EQ(promoted->size(), 64u);

  std::vector<uint8_t> readback = read_to_host(rh.get_ptr(), 64, stream1_);
  for (size_t i = 0; i < 64; ++i) {
    EXPECT_EQ(readback[i], static_cast<uint8_t>(0xA0 + (i % 16)));
  }
}

TEST_F(HipBufferTest, FromBuffer_PromotesCpuBufferForWrite)
{
  rosidl::Buffer<uint8_t> buffer(64);
  hip_buffer_backend::WriteHandle wh =
    hip_buffer_backend::from_output_buffer(buffer, stream1_);
  auto promoted = wh.get_promoted_buffer();
  ASSERT_NE(nullptr, promoted);
  EXPECT_EQ(promoted->get_backend_type(), "hip");
  EXPECT_EQ(promoted->size(), 64u);
  EXPECT_NE(nullptr, wh.get_ptr());

  write_pattern(wh.get_ptr(), 64, 123, stream1_);
}

TEST_F(HipBufferTest, FromBuffer_ThrowsOnEmptyBuffer)
{
  rosidl::Buffer<uint8_t> buffer;
  const rosidl::Buffer<uint8_t> & cbuf = buffer;

  EXPECT_THROW(
    hip_buffer_backend::from_input_buffer(cbuf, stream1_),
    hip_buffer_backend::HipError);
}

TEST_F(HipBufferTest, ToBuffer_CopiesThroughWriteHandle)
{
  rosidl::Buffer<uint8_t> buffer;
  allocate_buffer(buffer, 64);

  std::vector<uint8_t> host_data(64, 0xCD);
  {
    hip_buffer_backend::WriteHandle wh =
      hip_buffer_backend::from_output_buffer(buffer, stream1_);
    hip_buffer_backend::to_buffer(
      host_data.data(), 64, wh, stream1_, hipMemcpyHostToDevice);
  }

  const auto & cbuf = buffer;
  auto rh = hip_buffer_backend::from_input_buffer(cbuf, stream1_);
  std::vector<uint8_t> readback = read_to_host(rh.get_ptr(), 64, stream1_);
  EXPECT_EQ(readback[0], 0xCD);
  EXPECT_EQ(readback[63], 0xCD);
}

TEST_F(HipBufferTest, EventSync_WriteOnStream1_ReadOnStream2)
{
  constexpr size_t N = 1024;
  rosidl::Buffer<uint8_t> buffer;
  allocate_buffer(buffer, N);

  {
    hip_buffer_backend::WriteHandle wh =
      hip_buffer_backend::from_output_buffer(buffer, stream1_);
    write_pattern(wh.get_ptr(), N, 42, stream1_);
  }

  std::vector<uint8_t> result;
  {
    const rosidl::Buffer<uint8_t> & cbuf = buffer;
    hip_buffer_backend::ReadHandle rh =
      hip_buffer_backend::from_input_buffer(cbuf, stream2_);
    result = read_to_host(rh.get_ptr(), N, stream2_);
  }

  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(static_cast<uint8_t>((42 + i) % 256), result[i])
      << "Mismatch at index " << i;
  }
}

TEST_F(HipBufferTest, DoubleWriteHandle_Throws)
{
  rosidl::Buffer<uint8_t> buffer;
  allocate_buffer(buffer, 256);

  hip_buffer_backend::WriteHandle wh =
    hip_buffer_backend::from_output_buffer(buffer, stream1_);

  EXPECT_THROW(
    hip_buffer_backend::from_output_buffer(buffer, stream2_),
    hip_buffer_backend::HipError);
}

TEST_F(HipBufferTest, WriteAfterFinalized_Throws)
{
  rosidl::Buffer<uint8_t> buffer;
  allocate_buffer(buffer, 256);

  {
    hip_buffer_backend::WriteHandle wh =
      hip_buffer_backend::from_output_buffer(buffer, stream1_);
  }

  EXPECT_THROW(
    hip_buffer_backend::from_output_buffer(buffer, stream1_),
    hip_buffer_backend::HipError);
}

TEST_F(HipBufferTest, ReadAfterReadEvents_BlocksWrite)
{
  rosidl::Buffer<uint8_t> buffer;
  allocate_buffer(buffer, 256);

  {
    hip_buffer_backend::WriteHandle wh =
      hip_buffer_backend::from_output_buffer(buffer, stream1_);
  }

  {
    const rosidl::Buffer<uint8_t> & cbuf = buffer;
    hip_buffer_backend::ReadHandle rh =
      hip_buffer_backend::from_input_buffer(cbuf, stream2_);
  }

  EXPECT_THROW(
    hip_buffer_backend::from_output_buffer(buffer, stream1_),
    hip_buffer_backend::HipError);
}

TEST_F(HipBufferTest, Clone_PreservesData)
{
  constexpr size_t N = 512;
  rosidl::Buffer<uint8_t> buffer;
  allocate_buffer(buffer, N);

  {
    hip_buffer_backend::WriteHandle wh =
      hip_buffer_backend::from_output_buffer(buffer, stream1_);
    write_pattern(wh.get_ptr(), N, 77, stream1_);
  }

  auto * impl = const_cast<hip_buffer_backend::HipBufferImpl<uint8_t> *>(
    dynamic_cast<const hip_buffer_backend::HipBufferImpl<uint8_t> *>(buffer.get_impl()));
  impl->set_stream(nullptr);

  auto cloned_impl = impl->clone();
  auto * cloned_hip = dynamic_cast<hip_buffer_backend::HipBufferImpl<uint8_t> *>(
    cloned_impl.get());
  ASSERT_NE(nullptr, cloned_hip);

  const auto & cbuf = cloned_hip->get_hip_buffer();
  hip_buffer_backend::ReadHandle rh = cbuf.get_read_handle(stream2_);
  std::vector<uint8_t> result = read_to_host(rh.get_ptr(), N, stream2_);

  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(static_cast<uint8_t>((77 + i) % 256), result[i])
      << "Clone mismatch at index " << i;
  }
}

TEST_F(HipBufferTest, ToCpu_PreservesData)
{
  constexpr size_t N = 1024;
  rosidl::Buffer<uint8_t> buffer;
  allocate_buffer(buffer, N);

  {
    hip_buffer_backend::WriteHandle wh =
      hip_buffer_backend::from_output_buffer(buffer, stream1_);
    write_pattern(wh.get_ptr(), N, 55, stream1_);
  }

  auto * impl = const_cast<hip_buffer_backend::HipBufferImpl<uint8_t> *>(
    dynamic_cast<const hip_buffer_backend::HipBufferImpl<uint8_t> *>(buffer.get_impl()));
  impl->set_stream(nullptr);

  auto cpu_impl = buffer.get_impl()->to_cpu();
  ASSERT_NE(nullptr, cpu_impl);

  auto * cpu = dynamic_cast<rosidl::CpuBufferImpl<uint8_t> *>(cpu_impl.get());
  ASSERT_NE(nullptr, cpu);
  ASSERT_EQ(N, cpu->get_storage().size());

  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(static_cast<uint8_t>((55 + i) % 256), cpu->get_storage()[i])
      << "to_cpu mismatch at index " << i;
  }
}

TEST_F(HipBufferTest, Resize_PreservesPrefix)
{
  constexpr size_t N = 512;
  rosidl::Buffer<uint8_t> buffer;
  allocate_buffer(buffer, N);

  {
    hip_buffer_backend::WriteHandle wh =
      hip_buffer_backend::from_output_buffer(buffer, stream1_);
    write_pattern(wh.get_ptr(), N, 88, stream1_);
  }

  auto * impl = const_cast<hip_buffer_backend::HipBufferImpl<uint8_t> *>(
    dynamic_cast<const hip_buffer_backend::HipBufferImpl<uint8_t> *>(buffer.get_impl()));
  impl->set_stream(nullptr);
  impl->resize(N * 2);

  EXPECT_EQ(N * 2, impl->size());

  const auto & cbuf = impl->get_hip_buffer();
  hip_buffer_backend::ReadHandle rh = cbuf.get_read_handle(stream2_);
  std::vector<uint8_t> result = read_to_host(rh.get_ptr(), N, stream2_);

  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(static_cast<uint8_t>((88 + i) % 256), result[i])
      << "Resize prefix mismatch at index " << i;
  }
}

TEST_F(HipBufferTest, GpuPipeline_NoIntermediateCpuSync)
{
  constexpr size_t N = 4096;
  rosidl::Buffer<uint8_t> src_buffer;
  rosidl::Buffer<uint8_t> dst_buffer;
  allocate_buffer(src_buffer, N);
  allocate_buffer(dst_buffer, N);

  {
    hip_buffer_backend::WriteHandle wh =
      hip_buffer_backend::from_output_buffer(src_buffer, stream1_);
    write_pattern(wh.get_ptr(), N, 99, stream1_);
  }

  {
    const rosidl::Buffer<uint8_t> & cbuf = src_buffer;
    hip_buffer_backend::ReadHandle rh =
      hip_buffer_backend::from_input_buffer(cbuf, stream2_);
    hip_buffer_backend::WriteHandle wh2 =
      hip_buffer_backend::from_output_buffer(dst_buffer, stream2_);
    hipMemcpyAsync(
      wh2.get_ptr(), rh.get_ptr(), N, hipMemcpyDeviceToDevice, stream2_);
  }

  std::vector<uint8_t> result;
  {
    const rosidl::Buffer<uint8_t> & cbuf = dst_buffer;
    hip_buffer_backend::ReadHandle rh =
      hip_buffer_backend::from_input_buffer(cbuf, stream1_);
    result = read_to_host(rh.get_ptr(), N, stream1_);
  }

  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(static_cast<uint8_t>((99 + i) % 256), result[i])
      << "Mismatch at index " << i;
  }
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
