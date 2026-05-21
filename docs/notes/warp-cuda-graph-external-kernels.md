# Warp CUDA graph with external kernels

Conclusion: NVIDIA Warp CUDA graph capture can include external CUDA kernels, as long as those kernels launch on the same CUDA stream that is being captured and are CUDA graph-capture safe.

## Warp-only capture

```python
import warp as wp

@wp.kernel
def step(x: wp.array[float]):
    i = wp.tid()
    x[i] += 1.0

device = "cuda:0"
n = 1024
x = wp.zeros(n, dtype=float, device=device)

# Warm up/load outside capture to avoid lazy compilation or allocation during capture.
wp.load_module(device=device)

with wp.ScopedCapture(device=device, force_module_load=False) as capture:
    for _ in range(10):
        wp.launch(step, dim=n, inputs=[x], device=device)

graph = capture.graph

for _ in range(100):
    wp.capture_launch(graph)
```

## Include an external CUDA kernel

```python
stream = wp.Stream(device)

with wp.ScopedStream(stream):
    with wp.ScopedCapture(device=device, stream=stream, force_module_load=False) as capture:
        wp.launch(step, dim=n, inputs=[x], device=device, stream=stream)

        # External library / pybind / C++ launcher must use this cudaStream_t.
        external_lib.launch_my_kernel(
            x.ptr,
            n,
            int(stream.cuda_stream),
        )

        wp.launch(step, dim=n, inputs=[x], device=device, stream=stream)

graph = capture.graph
wp.capture_launch(graph, stream=stream)
```

## External capture owner, e.g. PyTorch CUDAGraph

If another framework starts the CUDA graph capture, use `external=True` so Warp joins the already active capture instead of starting its own.

```python
import torch
import warp as wp

device = "cuda:0"
torch_device = wp.device_to_torch(device)

t = torch.zeros(1024, device=torch_device)
a = wp.from_torch(t)

g = torch.cuda.CUDAGraph()
torch_stream = torch.cuda.Stream(device=torch_device)
warp_stream = wp.stream_from_torch(torch_stream)

wp.load_module(device=device)

with wp.ScopedStream(warp_stream), torch.cuda.graph(g, stream=torch_stream):
    wp.capture_begin(stream=warp_stream, force_module_load=False, external=True)
    try:
        wp.launch(step, dim=t.numel(), inputs=[a], device=device, stream=warp_stream)
    finally:
        wp.capture_end(stream=warp_stream)

g.replay()
```

## Rules and limitations

- External kernels must launch on the captured stream, e.g. `stream.cuda_stream`.
- Do not let external code silently use the legacy default stream or unrelated internal streams unless synchronization is deliberately represented in the capture.
- Warm up external libraries outside capture to avoid lazy initialization, compilation, allocation, or synchronization inside capture.
- Avoid capture-unsafe CUDA calls during capture, such as blocking synchronization or unsupported allocation patterns.
- Graph structure is fixed after capture. Kernel sequence, grid/block shape, and captured argument addresses are effectively fixed for replay.
- Prefer reusing stable buffers and updating their contents rather than replacing arrays/tensors with new allocations.
- Native in-process CUDA graph replay can include external kernels captured on the stream. Warp APIC serialization (`apic=True`, `capture_save`) should not be assumed to serialize arbitrary external kernels.

## References

- Warp runtime graph capture APIs: `wp.capture_begin()`, `wp.capture_end()`, `wp.capture_launch()`, `wp.ScopedCapture()`.
- CUDA stream capture APIs: `cudaStreamBeginCapture` / `cudaStreamEndCapture` or driver equivalents.
