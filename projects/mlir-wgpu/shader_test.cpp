// Copyright 2019 The Dawn Authors
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

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <emscripten.h>
#include <emscripten/html5.h>
#include <iostream>
#include <memory>
#include <webgpu/webgpu_cpp.h>

static wgpu::Instance instance;
static wgpu::Device device;
static wgpu::Queue queue;
static wgpu::RenderPipeline pipeline;
static constexpr uint32_t kBindGroupOffset = 256;
static wgpu::BindGroup bindgroup;
static int testsStarted = 0;
static int testsCompleted = 0;

wgpu::Surface surface;
wgpu::TextureView canvasDepthStencilView;
const uint32_t kWidth = 300;
const uint32_t kHeight = 150;

wgpu::Device GetDevice(wgpu::DeviceDescriptor *descriptor) {
  wgpu::RequestAdapterWebXROptions xrOptions = {};
  wgpu::RequestAdapterOptions options = {};
  options.nextInChain = &xrOptions;

  wgpu::Adapter adapter;
  wgpu::Future f1 = instance.RequestAdapter(
      &options, wgpu::CallbackMode::WaitAnyOnly,
      [&](wgpu::RequestAdapterStatus status, wgpu::Adapter ad,
          wgpu::StringView message) {
        if (message.length) {
          printf("RequestAdapter: %.*s\n", (int)message.length, message.data);
        }
        if (status == wgpu::RequestAdapterStatus::Unavailable) {
          printf("WebGPU unavailable; exiting cleanly\n");
          // exit(0) (rather than emscripten_force_exit(0)) ensures there is no
          // dangling keepalive.
          exit(0);
        }
        assert(status == wgpu::RequestAdapterStatus::Success);

        adapter = std::move(ad);
      });
  instance.WaitAny(f1, UINT64_MAX);
  assert(adapter);

  wgpu::Device device;
  wgpu::Future f2 = adapter.RequestDevice(
      descriptor, wgpu::CallbackMode::WaitAnyOnly,
      [&](wgpu::RequestDeviceStatus status, wgpu::Device dev,
          wgpu::StringView message) {
        if (message.length) {
          printf("RequestDevice: %.*s\n", (int)message.length, message.data);
        }
        assert(status == wgpu::RequestDeviceStatus::Success);

        device = std::move(dev);
      });
  instance.WaitAny(f2, UINT64_MAX);
  assert(device);

  return device;
}

static const char shaderCode[] = R"(
    @binding(0) @group(0) var<uniform> uColor : vec4f;

    @vertex
    fn main_v(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
        var pos = array<vec2<f32>, 3>(
            vec2<f32>(0.0, 0.5), vec2<f32>(-0.5, -0.5), vec2<f32>(0.5, -0.5));
        return vec4<f32>(pos[idx], 0.0, 1.0);
    }
    @fragment
    fn main_f() -> @location(0) vec4<f32> {
        return uColor;
    }
)";

void init() {
  queue = device.GetQueue();

  // Test of OOM with mappedAtCreation.
  {
    wgpu::BufferDescriptor descriptor{};
    descriptor.usage = wgpu::BufferUsage::CopyDst;
    descriptor.size = 0x10'0000'0000'0000ULL;
    descriptor.mappedAtCreation = true;
    wgpu::Buffer bufferTooLarge = device.CreateBuffer(&descriptor);
    assert(bufferTooLarge == nullptr);
  }

  wgpu::ShaderModule shaderModule{};
  {
    wgpu::ShaderSourceWGSL wgslDesc{};
    wgslDesc.code = shaderCode;

    wgpu::ShaderModuleDescriptor descriptor{};
    descriptor.nextInChain = &wgslDesc;
    shaderModule = device.CreateShaderModule(&descriptor);
  }

  wgpu::BindGroupLayout bgl;
  {
    wgpu::BindGroupLayoutEntry bglEntry{
        .binding = 0,
        .visibility = wgpu::ShaderStage::Fragment,
        //.bindingArraySize = 1,
        .buffer =
            {
                .type = wgpu::BufferBindingType::Uniform,
                .hasDynamicOffset = true,
            },
    };
    wgpu::BindGroupLayoutDescriptor bglDesc{
        .entryCount = 1,
        .entries = &bglEntry,
    };
    bgl = device.CreateBindGroupLayout(&bglDesc);

    static constexpr std::array<float, 4> kColor{0.0, 0.502, 1.0,
                                                 1.0}; // 0x80/0xff ~= 0.502
    wgpu::BufferDescriptor uniformBufferDesc{
        .usage = wgpu::BufferUsage::Uniform,
        .size = kBindGroupOffset + sizeof(kColor),
        .mappedAtCreation = true,
    };
    wgpu::Buffer uniformBuffer = device.CreateBuffer(&uniformBufferDesc);
    {
      float *mapped = reinterpret_cast<float *>(
          uniformBuffer.GetMappedRange(kBindGroupOffset));
      memcpy(mapped, kColor.data(), sizeof(kColor));
      uniformBuffer.Unmap();
    }

    wgpu::BindGroupEntry bgEntry{
        .binding = 0,
        .buffer = uniformBuffer,
        .size = sizeof(kColor),
    };
    wgpu::BindGroupDescriptor bgDesc{
        .layout = bgl,
        .entryCount = 1,
        .entries = &bgEntry,
    };
    bindgroup = device.CreateBindGroup(&bgDesc);
  }

  {
    wgpu::PipelineLayoutDescriptor pl{};
    pl.bindGroupLayoutCount = 1;
    pl.bindGroupLayouts = &bgl;

    wgpu::ColorTargetState colorTargetState{};
    colorTargetState.format = wgpu::TextureFormat::BGRA8Unorm;

    wgpu::FragmentState fragmentState{};
    fragmentState.module = shaderModule;
    fragmentState.entryPoint = "main_f";
    fragmentState.targetCount = 1;
    fragmentState.targets = &colorTargetState;

    wgpu::DepthStencilState depthStencilState{};
    depthStencilState.format = wgpu::TextureFormat::Depth32Float;
    depthStencilState.depthWriteEnabled = true;
    depthStencilState.depthCompare = wgpu::CompareFunction::Always;

    wgpu::RenderPipelineDescriptor descriptor{};
    descriptor.layout = device.CreatePipelineLayout(&pl);
    descriptor.vertex.module = shaderModule;
    descriptor.vertex.entryPoint = "main_v";
    descriptor.fragment = &fragmentState;
    descriptor.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    descriptor.depthStencil = &depthStencilState;

    // Just test the bindings; we are only going to actually use the async one
    // below.
    wgpu::RenderPipeline unused = device.CreateRenderPipeline(&descriptor);
    assert(unused);

    wgpu::Future f = device.CreateRenderPipelineAsync(
        &descriptor, wgpu::CallbackMode::WaitAnyOnly,
        [](wgpu::CreatePipelineAsyncStatus status, wgpu::RenderPipeline pl,
           wgpu::StringView message) {
          if (message.length) {
            printf("CreateRenderPipelineAsync: %.*s\n", (int)message.length,
                   message.data);
          }
          assert(status == wgpu::CreatePipelineAsyncStatus::Success);
          pipeline = std::move(pl);
        });
    instance.WaitAny(f, UINT64_MAX);
    assert(pipeline);
  }
}

// The depth stencil attachment isn't really needed to draw the triangle
// and doesn't really affect the render result.
// But having one should give us a slightly better test coverage for the compile
// of the depth stencil descriptor.
void render(wgpu::TextureView view, wgpu::TextureView depthStencilView) {
  wgpu::RenderPassColorAttachment attachment{};
  attachment.view = view;
  attachment.loadOp = wgpu::LoadOp::Clear;
  attachment.storeOp = wgpu::StoreOp::Store;
  attachment.clearValue = {0, 1, 0, 0.5};

  wgpu::RenderPassDescriptor renderpass{};
  renderpass.colorAttachmentCount = 1;
  renderpass.colorAttachments = &attachment;

  wgpu::RenderPassDepthStencilAttachment depthStencilAttachment = {};
  depthStencilAttachment.view = depthStencilView;
  depthStencilAttachment.depthClearValue = 0;
  depthStencilAttachment.depthLoadOp = wgpu::LoadOp::Clear;
  depthStencilAttachment.depthStoreOp = wgpu::StoreOp::Store;

  renderpass.depthStencilAttachment = &depthStencilAttachment;

  wgpu::CommandBuffer commands;
  {
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    {
      wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&renderpass);
      pass.SetPipeline(pipeline);
      pass.SetBindGroup(0, bindgroup, 1, &kBindGroupOffset);
      pass.Draw(3);
      pass.End();
    }
    commands = encoder.Finish();
  }

  queue.Submit(1, &commands);
}

void issueContentsCheck(const char *functionName, wgpu::Buffer readbackBuffer,
                        uint32_t expectData) {
  testsStarted++;
  wgpu::Future f = readbackBuffer.MapAsync(
      wgpu::MapMode::Read, 0, 4, wgpu::CallbackMode::WaitAnyOnly,
      [=](wgpu::MapAsyncStatus status, wgpu::StringView message) {
        if (message.length) {
          printf("issueContentsCheck MapAsync: %.*s\n", (int)message.length,
                 message.data);
        }
        assert(status == wgpu::MapAsyncStatus::Success);

        static constexpr bool kUseReadMappedRange = true;

        std::vector<char> ptrData;
        const void *ptr;
        if constexpr (kUseReadMappedRange) {
          ptrData.resize(4);
          ptr = ptrData.data();
          wgpu::Status status =
              readbackBuffer.ReadMappedRange(0, ptrData.data(), 4);
          printf("%s: ReadMappedRange -> %u%s\n", functionName, status,
                 status == wgpu::Status::Success ? "" : " <------- FAILED");
          assert(status == wgpu::Status::Success);
        } else {
          ptr = readbackBuffer.GetConstMappedRange();
          printf("%s: GetConstMappedRange -> %p%s\n", functionName, ptr,
                 ptr ? "" : " <------- FAILED");
          assert(ptr != nullptr);
        }

        uint32_t readback = static_cast<const uint32_t *>(ptr)[0];
        printf("  got %08x, expected %08x%s\n", readback, expectData,
               readback == expectData ? "" : " <------- FAILED");

        readbackBuffer.Unmap();
        testsCompleted++;
      });
  instance.WaitAny(f, UINT64_MAX);
}

void doCopyTestMappedAtCreation(bool useRange) {
  static constexpr uint32_t kValue = 0x05060708;
  size_t offset = useRange ? 8 : 0;
  size_t size = useRange ? 12 : 4;
  wgpu::Buffer src;
  {
    wgpu::BufferDescriptor descriptor{};
    descriptor.size = size;
    descriptor.usage = wgpu::BufferUsage::CopySrc;
    // descriptor.usage = static_cast<wgpu::BufferUsage>(0xffff'ffff); //
    // Uncomment to make createBuffer fail
    descriptor.mappedAtCreation = true;
    src = device.CreateBuffer(&descriptor);
    // Calls just to check they work
    src.GetSize();
    src.GetUsage();
  }

  static constexpr bool kUseWriteMappedRange = true;
  if constexpr (kUseWriteMappedRange) {
    wgpu::Status status = src.WriteMappedRange(offset, &kValue, 4);
    printf("%s: WriteMappedRange -> %u%s\n", __FUNCTION__, status,
           status == wgpu::Status::Success ? "" : " <------- FAILED");
    assert(status == wgpu::Status::Success);
  } else {
    uint32_t *ptr = static_cast<uint32_t *>(
        useRange ? src.GetMappedRange(offset, 4) : src.GetMappedRange());
    printf("%s: GetMappedRange -> %p%s\n", __FUNCTION__, ptr,
           ptr ? "" : " <------- FAILED");
    assert(ptr != nullptr);
    *ptr = kValue;
  }
  src.Unmap();

  wgpu::Buffer dst;
  {
    wgpu::BufferDescriptor descriptor{};
    descriptor.size = 4;
    descriptor.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    dst = device.CreateBuffer(&descriptor);
  }

  wgpu::CommandBuffer commands;
  {
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(src, offset, dst, 0, 4);
    commands = encoder.Finish();
  }
  queue.Submit(1, &commands);

  issueContentsCheck(__FUNCTION__, dst, kValue);
}

void doCopyTestMapAsync(bool useRange) {
  static constexpr uint32_t kValue = 0x01020304;
  size_t size = useRange ? 12 : 4;
  wgpu::Buffer src;
  {
    wgpu::BufferDescriptor descriptor{};
    descriptor.size = size;
    descriptor.usage = wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc;
    src = device.CreateBuffer(&descriptor);
  }
  size_t offset = useRange ? 8 : 0;

  const char *functionName = __FUNCTION__;
  wgpu::Future f = src.MapAsync(
      wgpu::MapMode::Write, offset, 4, wgpu::CallbackMode::AllowSpontaneous,
      [=](wgpu::MapAsyncStatus status, wgpu::StringView message) {
        if (message.length) {
          printf("doCopyTestMapAsync MapAsync: %.*s\n", (int)message.length,
                 message.data);
        }
        assert(status == wgpu::MapAsyncStatus::Success);

        uint32_t *ptr = static_cast<uint32_t *>(
            useRange ? src.GetMappedRange(offset, 4) : src.GetMappedRange());
        printf("%s: getMappedRange -> %p%s\n", functionName, ptr,
               ptr ? "" : " <------- FAILED");
        assert(ptr != nullptr);
        *ptr = kValue;
        src.Unmap();
      });
  instance.WaitAny(f, UINT64_MAX);

  // TODO: Doesn't work if this is inside the MapAsync callback because it
  // causes nested WaitAny to happen and crashes Emscripten.
  wgpu::Buffer dst;
  {
    wgpu::BufferDescriptor descriptor{};
    descriptor.size = 4;
    descriptor.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    dst = device.CreateBuffer(&descriptor);
  }

  wgpu::CommandBuffer commands;
  {
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(src, offset, dst, 0, 4);
    commands = encoder.Finish();
  }
  queue.Submit(1, &commands);

  issueContentsCheck(functionName, dst, kValue);
}

void doRenderTest() {
  wgpu::Texture readbackTexture;
  {
    wgpu::TextureDescriptor descriptor{};
    descriptor.usage =
        wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc;
    descriptor.size = {1, 1, 1};
    descriptor.format = wgpu::TextureFormat::BGRA8Unorm;
    readbackTexture = device.CreateTexture(&descriptor);
    // Calls just to check they work
    readbackTexture.GetWidth();
    readbackTexture.GetHeight();
    readbackTexture.GetDepthOrArrayLayers();
    readbackTexture.GetDimension();
    readbackTexture.GetFormat();
    readbackTexture.GetMipLevelCount();
    readbackTexture.GetSampleCount();
    readbackTexture.GetUsage();
  }
  wgpu::Texture depthTexture;
  {
    wgpu::TextureDescriptor descriptor{};
    descriptor.usage = wgpu::TextureUsage::RenderAttachment;
    descriptor.size = {1, 1, 1};
    descriptor.format = wgpu::TextureFormat::Depth32Float;
    depthTexture = device.CreateTexture(&descriptor);
  }
  render(readbackTexture.CreateView(), depthTexture.CreateView());

  wgpu::Buffer readbackBuffer;
  {
    wgpu::BufferDescriptor descriptor{};
    descriptor.size = 4;
    descriptor.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;

    readbackBuffer = device.CreateBuffer(&descriptor);
  }

  wgpu::CommandBuffer commands;
  {
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    wgpu::TexelCopyTextureInfo src{};
    src.texture = readbackTexture;
    src.origin = {0, 0, 0};
    wgpu::TexelCopyBufferInfo dst{};
    dst.buffer = readbackBuffer;
    dst.layout.bytesPerRow = 256;
    wgpu::Extent3D extent = {1, 1, 1};
    encoder.CopyTextureToBuffer(&src, &dst, &extent);
    commands = encoder.Finish();
  }
  queue.Submit(1, &commands);

  // Check the color value encoded in the shader makes it out correctly.
  static const uint32_t expectData = 0xff0080ff;
  issueContentsCheck(__FUNCTION__, readbackBuffer, expectData);
}

extern "C" {
EMSCRIPTEN_KEEPALIVE bool frame();
}

static int frameNum = 0;
bool frame() {
  frameNum++;
  if (frameNum == 1) {
    printf("Running frame-1 tests...\n");
    // Another copy of doRenderTest to make sure it works in the frame loop.
    // Note this function is async (via Asyncify/JSPI) so the SurfaceTexture
    // lifetime must not span it! (it may expire while waiting for mapAsync to
    // complete)
    doRenderTest();
    return true;
  }

  wgpu::SurfaceTexture surfaceTexture;
  surface.GetCurrentTexture(&surfaceTexture);
  assert(surfaceTexture.status ==
         wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal);
  wgpu::TextureView backbuffer = surfaceTexture.texture.CreateView();

  if (frameNum == 2) {
    printf("Running frame 2 and continuing!\n");
  }
  render(backbuffer, canvasDepthStencilView);
  // TODO: On frame 1, read back from the canvas with drawImage() (or something)
  // and check the result.

#if defined(__EMSCRIPTEN__)
  // Stop running after a few frames in Emscripten.
  if (frameNum >= 10 && testsCompleted == testsStarted) {
    printf("Several frames rendered and no pending tests remaining!\n");
    printf("Stopping main loop and destroying device to clean up.\n");
    device.Destroy();
    return false; // Stop the requestAnimationFrame loop
  }
#endif

  return true; // Continue the requestAnimationFrame loop (Wasm) or main loop
               // (native)
}

void run() {
  init();

  printf("Running startup tests...\n");

  // Kick off all of the tests before setting up to render a frame.
  // (Note we don't wait for the tests so they may complete before or after the
  // frame.)
  doCopyTestMappedAtCreation(false);
  doCopyTestMappedAtCreation(true);
  doCopyTestMapAsync(false);
  doCopyTestMapAsync(true);
  doRenderTest();

  {
    wgpu::TextureDescriptor descriptor{};
    descriptor.usage = wgpu::TextureUsage::RenderAttachment;
    descriptor.size = {kWidth, kHeight, 1};
    descriptor.format = wgpu::TextureFormat::Depth32Float;
    canvasDepthStencilView = device.CreateTexture(&descriptor).CreateView();
  }

  printf("Starting main loop...\n");
#if defined(__EMSCRIPTEN__)
  {
    wgpu::EmscriptenSurfaceSourceCanvasHTMLSelector canvasDesc{};
    canvasDesc.selector = "#canvas";

    wgpu::SurfaceDescriptor surfDesc{};
    surfDesc.nextInChain = &canvasDesc;
    surface = instance.CreateSurface(&surfDesc);

    wgpu::SurfaceColorManagement colorManagement{};
    wgpu::SurfaceConfiguration configuration{};
    configuration.nextInChain = &colorManagement;
    configuration.device = device;
    configuration.usage = wgpu::TextureUsage::RenderAttachment;
    configuration.format = wgpu::TextureFormat::BGRA8Unorm;
    configuration.width = kWidth;
    configuration.height = kHeight;
    configuration.alphaMode = wgpu::CompositeAlphaMode::Premultiplied;
    configuration.presentMode = wgpu::PresentMode::Fifo;
    surface.Configure(&configuration);
  }

  // Workaround for JSPI not working in emscripten_set_main_loop. Loosely based
  // on this code:
  // https://github.com/emscripten-core/emscripten/issues/22493#issuecomment-2330275282
  // Note the following link args are required:
  // - JSPI: -sDEFAULT_LIBRARY_FUNCS_TO_INCLUDE=$getWasmTableEntry
  // - Asyncify: -sEXPORTED_RUNTIME_METHODS=ccall
  // clang-format off
    EM_ASM({
#    if DEMO_USE_JSPI // -sJSPI=1 (aka -sASYNCIFY=2)
        var callback = WebAssembly.promising(getWasmTableEntry($0));
#    else // -sASYNCIFY=1
        // ccall seems to be the only thing in Emscripten which lets us turn an
        // Asyncified Wasm function into a JS function returning a Promise.
        // It can only call exported functions.
        var callback = () => globalThis['Module']['ccall']("frame", "boolean", [], [], {async: true});
#    endif // DEMO_USE_JSPI
        async function tick() {
            // Start the frame callback. 'await' means we won't call
            // requestAnimationFrame again until it completes.
            var keepLooping = await callback();
            if (keepLooping) requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);
    }, frame);
    // clang-format off
#endif
    printf("Stopping main loop and destroying device to clean up.\n");
}

int main() {
    printf("Initializing...\n");
    wgpu::InstanceDescriptor desc;
    static constexpr auto kTimedWaitAny = wgpu::InstanceFeatureName::TimedWaitAny;
    desc.requiredFeatureCount = 1;
    desc.requiredFeatures = &kTimedWaitAny;
    instance = wgpu::CreateInstance(&desc);

    {
        wgpu::Limits limits;
        //limits.maxBufferSize = 0xffff'ffff'ffffLLU; // Uncomment to make requestDevice fail
        //limits.maxImmediateSize = 1; // Uncomment to test maxImmediateSize passthrough
        wgpu::DeviceDescriptor desc;
        desc.requiredLimits = &limits;
        desc.SetUncapturedErrorCallback(
            [](const wgpu::Device&, wgpu::ErrorType errorType, wgpu::StringView message) {
                printf("UncapturedError (errorType=%d): %.*s\n", errorType, (int)message.length, message.data);
                assert(false);
            });
        desc.SetDeviceLostCallback(wgpu::CallbackMode::AllowSpontaneous,
            [](const wgpu::Device&, wgpu::DeviceLostReason reason, wgpu::StringView message) {
                printf("DeviceLost (reason=%d): %.*s\n", reason, (int)message.length, message.data);
            });
        device = GetDevice(&desc);
    }

    run();

    // The test result will be reported when the main_loop completes.
    // emscripten_exit_with_live_runtime() shouldn't be needed, because the async stuff we do keeps
    // the runtime alive automatically. (Note the tests may complete before or after the frame.)
    // - The WebGPU callbacks keep the runtime alive until they complete.
    // - emscripten_set_main_loop keeps it alive until emscripten_cancel_main_loop.
    //
    // This code is returned when the runtime exits unless something else sets it, like exit(0).
    return 99;
}
