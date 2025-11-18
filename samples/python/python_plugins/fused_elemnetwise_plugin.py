import tensorrt as trt
import torch
import triton
import triton.language as tl
from typing import List
import cupy
import numpy as np

def get_cuda_device():
    if torch.cuda.is_available():
        current_gpu_idx = torch.cuda.current_device()
        device = torch.device(f"cuda:{current_gpu_idx}")
        return device
    else:
        raise RuntimeError("No CUDA device available for Triton kernel")

def get_full_shape(data_list):
    if len(data_list) == 0:
        return None

    first_shape = data_list[0].shape
    full_shape = [None] * len(first_shape)
    num_dims = len(first_shape)
    for data in data_list:
        data_shape = data.shape

        if len(data_shape) != num_dims:
            print("dismatch rank: ", data_shape, ", demand: ", num_dims)
            return None

        for i in range(num_dims):
            if full_shape[i] is None:
                full_shape[i] = data_shape[i]
            else:
                if full_shape[i] == 1:
                    if data_shape[i] == 1:
                        continue
                    else:
                        full_shape[i] == data_shape[i]
                else:
                    if data_shape[i] == 1:
                        continue
                    if data_shape[i] != full_shape[i]:
                        print(
                            "full shape: ",
                            full_shape,
                            ", current shape: ",
                            data_shape,
                            ", dismatch dim at: ",
                            i,
                        )
                        return None

    full_strides = [None] * num_dims
    full_strides[-1] = 1
    for i in range(num_dims - 1, 0, -1):
        full_strides[i - 1] = full_strides[i] * full_shape[i]

    strides = [None] * len(data_list)

    for k, data in enumerate(data_list):
        data_stride = list(data.stride())
        data_shape = data.shape
        for i in range(num_dims):
            if data_shape[i] == 1:
                data_stride[i] = 0
        strides[k] = tuple(data_stride)

    return full_shape, tuple(full_strides), strides


@triton.jit
def ReLU(x):
    return tl.where(x >= 0, x, 0)

'''
# If your trition is newer than 3.3, you can use the following function to project the coordinate.
# Newer version of triton could support tuple as constexpr, older only support scalar
@triton.jit
def coordinate_project(
        idx,
        num_dims: tl.constexpr,
        full_strides: tl.constexpr,
        broadcast_strides: tl.constexpr,
):
    broadcast_offset = tl.zeros_like(idx)
    for stride_idx in tl.static_range(num_dims):
        full_stride = full_strides[stride_idx]
        broadcast_stride = broadcast_strides[stride_idx]
        pre_quotient = idx // full_stride
        idx = idx % full_stride
        broadcast_offset += pre_quotient * broadcast_stride
    return broadcast_offset
    
@triton.jit
def broadcast_elementwise_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    output_ptr,
    n_elements,
    num_dims: tl.constexpr,
    full_strides: tl.constexpr,
    broadcast_strides: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    output_offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = output_offset < n_elements

    broadcast_offset = coordinate_project(
        output_offset, num_dims, full_strides, broadcast_strides
    )

    x = tl.load(x_ptr + output_offset, mask=mask)
    y = tl.load(y_ptr + output_offset, mask=mask)
    z = tl.load(z_ptr + broadcast_offset, mask=mask)
    output = ReLU(x + y - z)
    tl.store(output_ptr + output_offset, output, mask=mask)
'''

# This type of coordinate_project only support fixed rank
@triton.jit
def coordinate_project(
        idx,
        num_dims: tl.constexpr,
        full_strides_0: tl.constexpr,
        full_strides_1: tl.constexpr,
        full_strides_2: tl.constexpr,
        broadcast_strides_0: tl.constexpr,
        broadcast_strides_1: tl.constexpr,
        broadcast_strides_2: tl.constexpr,
):
    broadcast_offset = tl.zeros_like(idx)

    q1 = idx // full_strides_0
    idx = idx % full_strides_0
    broadcast_offset += q1 * broadcast_strides_0
    q2 = idx // full_strides_1
    idx = idx % full_strides_1
    broadcast_offset += q2 * broadcast_strides_1
    q3 = idx // full_strides_2
    broadcast_offset += q3 * broadcast_strides_2
    return broadcast_offset

@triton.jit
def broadcast_elementwise_kernel(
        x_ptr,
        y_ptr,
        z_ptr,
        output_ptr,
        n_elements,
        num_dims: tl.constexpr,
        full_strides_0: tl.constexpr,
        full_strides_1: tl.constexpr,
        full_strides_2: tl.constexpr,
        broadcast_strides_0: tl.constexpr,
        broadcast_strides_1: tl.constexpr,
        broadcast_strides_2: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    output_offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = output_offset < n_elements

    broadcast_offset = coordinate_project(
        output_offset, num_dims,
        full_strides_0, full_strides_1, full_strides_2,
        broadcast_strides_0, broadcast_strides_1, broadcast_strides_2
    )

    x = tl.load(x_ptr + output_offset, mask=mask)
    y = tl.load(y_ptr + output_offset, mask=mask)
    z = tl.load(z_ptr + broadcast_offset, mask=mask)
    output = ReLU(x + y - z)
    tl.store(output_ptr + output_offset, output, mask=mask)


def broadcast_elementwise(
        x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == len(y.shape) == len(z.shape) == len(output.shape) == 3
    assert x.shape == y.shape
    full_shape, full_strides, strides = get_full_shape([x, y, z])
    broadcast_strides = strides[-1]
    num_dims = len(full_shape)
    assert (
            x.device ==
            y.device ==
            z.device ==
            output.device
    )
    num_elements = output.numel()
    grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)
    broadcast_elementwise_kernel[grid](
        x,
        y,
        z,
        output,
        num_elements,
        num_dims,
        full_strides[0],
        full_strides[1],
        full_strides[2],
        broadcast_strides[0],
        broadcast_strides[1],
        broadcast_strides[2],
        BLOCK_SIZE=256,
    )

'''

# Kernel test code
def precision_test(data_type):
    DEVICE = get_cuda_device()
    full_shape = [256, 512, 1000]
    broadcast_idx = [1]
    broadcast_shape = [1 if i in broadcast_idx else x for i, x in enumerate(full_shape)]
    torch.manual_seed(42)

    x = torch.rand(full_shape, device=DEVICE, dtype=data_type)
    y = torch.rand(full_shape, device=DEVICE, dtype=data_type)
    z = torch.rand(broadcast_shape, device=DEVICE, dtype=data_type)
    output_triton = torch.rand(full_shape, device=DEVICE, dtype=data_type)

    output_torch = torch.relu(x + y - z)
    broadcast_elementwise(x, y, z, output_triton)
    print("Precision test for data type: ", data_type)
    print(
        f"The maximum difference of result between torch and triton is "
        f"{torch.max(torch.abs(output_torch - output_triton))}"
    )


# precision_test(torch.float32)

# precision_test(torch.float16)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["data_size"],  # Argument names to use as an x-axis for the plot.
        x_vals=[
            2**i for i in range(5, 15, 1)
        ],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg="backend",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=["triton", "torch"],  # Possible values for `line_arg`.
        line_names=["Triton", "Torch"],  # Label name for the lines.
        styles=[("blue", "-"), ("green", "-")],  # Line styles.
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name="braodcast-elementwise-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={
            "M": 256,
            "K": 512,
            "data_type": torch.float16,
        },  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(data_size, M, K, backend, data_type):
    DEVICE = get_cuda_device()
    full_shape = [M, K, data_size]
    broadcast_shape = [M, 1, data_size]
    x = torch.rand(full_shape, device=DEVICE, dtype=data_type)
    y = torch.rand(full_shape, device=DEVICE, dtype=data_type)
    z = torch.rand(broadcast_shape, device=DEVICE, dtype=data_type)
    output_triton = torch.rand(full_shape, device=DEVICE, dtype=data_type)
    quantiles = [0.5, 0.1, 0.9]
    if backend == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.relu(x + y - z), quantiles=quantiles
        )
    if backend == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: broadcast_elementwise(x, y, z, output_triton), quantiles=quantiles
        )

    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

# benchmark.run(print_data=True, show_plots=True)

'''
def get_torch_dtype(input_dtype: trt.DataType) -> torch.dtype:
    if input_dtype == trt.float32:
        return torch.float32
    elif input_dtype == trt.float16:
        return torch.float16
    else:
        assert False

class FusedElementwisePlugin(trt.IPluginV2DynamicExt):
    def __init__(self: trt.IPluginV2DynamicExt):
        trt.IPluginV2DynamicExt.__init__(self)
        self.plugin_namespace = ""
        self.plugin_name = "FusedElementwisePlugin"
        self.plugin_version = "1"
        self.plugin_type = self.plugin_name
        self.num_outputs = 1
        self.cuda_device = get_cuda_device()
        self.original_kernel = broadcast_elementwise_kernel
        self.optimized_kernel= None
        self.optimized_config = {}
        self.block_size = 256

    def clone(self) -> trt.IPluginV2DynamicExt:
        cloned_plugin = FusedElementwisePlugin()
        cloned_plugin.__dict__.update(self.__dict__)
        return cloned_plugin

    def get_workspace_size(self, dptd_in: List[trt.PluginTensorDesc], dptd_out: List[trt.PluginTensorDesc]) -> int:
        return 0

    def get_output_datatype(self: trt.IPluginV2DynamicExt, index: int, input_types: List[trt.DataType]) -> trt.DataType:
        return input_types[0]

    def get_output_dimensions(self, output_index, inputs, exprBuilder):
        return inputs[0]

    def supports_format_combination(self, pos: int, in_out: List[trt.PluginTensorDesc], num_inputs: int) -> bool:
        desc = in_out[pos]
        if pos == 0:
            res = (desc.type == trt.float32 or desc.type == trt.float16) and desc.format == trt.TensorFormat.LINEAR
        else:
            res = desc.format == trt.TensorFormat.LINEAR and desc.type == in_out[0].type
        return res

    def configure_plugin(self, inputs, outputs):
        # Build phase miss max, opt, min
        if len(inputs[0].desc.dims) == 0:
            return

        torch_dtype = get_torch_dtype(inputs[0].desc.type)
        input_1_tensor = torch.rand(list(inputs[0].desc.dims), device=self.cuda_device, dtype=torch_dtype)
        input_2_tensor = torch.rand(list(inputs[1].desc.dims), device=self.cuda_device, dtype=torch_dtype)
        input_3_tensor = torch.rand(list(inputs[2].desc.dims), device=self.cuda_device, dtype=torch_dtype)
        output_tensor = torch.rand(list(outputs[0].desc.dims), device=self.cuda_device, dtype=torch_dtype)
        full_shape, full_strides, strides = get_full_shape([input_1_tensor, input_2_tensor, input_3_tensor])
        broadcast_strides = strides[-1]
        num_dims = len(full_shape)
        num_elements = output_tensor.numel()
        self.optimized_kernel = self.original_kernel.warmup(input_1_tensor,
                                                            input_2_tensor,
                                                            input_3_tensor,
                                                            output_tensor,
                                                            num_elements,
                                                            num_dims,
                                                            full_strides[0],
                                                            full_strides[1],
                                                            full_strides[2],
                                                            broadcast_strides[0],
                                                            broadcast_strides[1],
                                                            broadcast_strides[2],
                                                            BLOCK_SIZE=self.block_size, grid=(1, ))


    def get_serialization_size(self):
        return 0

    def get_torch_tensor_from_pointer(self, desc: trt.PluginTensorDesc, ptr: int) -> torch.Tensor:
        torch_dtype = get_torch_dtype(desc.type)
        np_dtype = trt.nptype(desc.type)
        wrapped_memory = cupy.cuda.UnownedMemory(ptr, np.prod(desc.dims) * np.dtype(np_dtype).itemsize, self)
        wrapped_ptr = cupy.cuda.MemoryPointer(wrapped_memory, 0)
        wrapped_ndarray = cupy.ndarray(desc.dims, dtype=np_dtype, memptr=wrapped_ptr)
        return torch.as_tensor(wrapped_ndarray, device=self.cuda_device, dtype=torch_dtype)

    def enqueue(self, input_desc: List[trt.PluginTensorDesc], output_desc: List[trt.PluginTensorDesc], inputs: List[int], outputs: List[int], workspace: int, stream: int) -> None:
        """ Execute kernel on GPU

        output = ReLU(input_0 + input_1 - input_2)

        """

        input_1_tensor = self.get_torch_tensor_from_pointer(input_desc[0], inputs[0])
        input_2_tensor = self.get_torch_tensor_from_pointer(input_desc[1], inputs[1])
        input_3_tensor = self.get_torch_tensor_from_pointer(input_desc[2], inputs[2])
        output_tensor = self.get_torch_tensor_from_pointer(output_desc[0], outputs[0])

        full_shape, full_strides, strides = get_full_shape([input_1_tensor, input_2_tensor, input_3_tensor])
        broadcast_strides = strides[-1]
        num_dims = len(full_shape)
        num_elements = output_tensor.numel()

        with torch.cuda.stream(torch.cuda.ExternalStream(stream, self.cuda_device)):
            if self.optimized_kernel is None:
                grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)
                self.original_kernel[grid](
                    input_1_tensor,
                    input_2_tensor,
                    input_3_tensor,
                    output_tensor,
                    num_elements,
                    num_dims,
                    full_strides[0],
                    full_strides[1],
                    full_strides[2],
                    broadcast_strides[0],
                    broadcast_strides[1],
                    broadcast_strides[2],
                    BLOCK_SIZE=self.block_size
                )
            else:
                self.optimized_kernel[(triton.cdiv(num_elements, self.block_size), 1, 1)](
                    input_1_tensor,
                    input_2_tensor,
                    input_3_tensor,
                    output_tensor,
                    num_elements,
                )

class FusedElementwisePluginCreator(trt.IPluginCreator):

    def __init__(self):
        trt.IPluginCreator.__init__(self)
        self.name = "FusedElementwisePlugin"
        self.plugin_version = "1"
        self.plugin_namespace = ""
        self.field_names = trt.PluginFieldCollection([])
        return

    def create_plugin(self, name: str, field_collection: trt.PluginFieldCollection):
        return FusedElementwisePlugin()

    def deserialize_plugin(self, name: str, serialized_plugin):
        return FusedElementwisePlugin()

def unittest():
    assert torch.cuda.is_available()
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    config = builder.create_builder_config()
    plugin_registry = trt.get_plugin_registry()
    plugin_creator = FusedElementwisePluginCreator()
    r_status = plugin_registry.register_creator(plugin_creator, "")
    assert r_status
    # print("register status: ", r_status)
    # registry_list = plugin_registry.plugin_creator_list
    # print("registry_list: ")
    # for registry in registry_list:
    #     print(registry.name, registry.plugin_version, registry.plugin_namespace)
    registered_creator = plugin_registry.get_plugin_creator("FusedElementwisePlugin", "1", "")
    assert registered_creator is not None
    full_shape = [256, 512, 1000]
    broadcast_idx = [1]
    broadcast_shape = [1 if i in broadcast_idx else x for i, x in enumerate(full_shape)]
    input_1 = network.add_input("input_1", trt.DataType.FLOAT, full_shape)
    input_2 = network.add_input("input_2", trt.DataType.FLOAT, full_shape)
    input_3 = network.add_input("input_3", trt.DataType.FLOAT, broadcast_shape)

    field_collection = trt.PluginFieldCollection([])
    plugin_object = registered_creator.create_plugin("FusedElementwise", field_collection)
    plugin_layer = network.add_plugin_v2([input_1, input_2, input_3], plugin_object)
    plugin_output = plugin_layer.get_output(0)
    plugin_output.name = "output"
    network.mark_output(plugin_output)
    plan = builder.build_serialized_network(network, config)


    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()

    torch.manual_seed(42)
    data_type = torch.float32
    DEVICE = get_cuda_device()
    input_1_gpu = torch.rand(full_shape, device=DEVICE, dtype=data_type)
    input_2_gpu = torch.rand(full_shape, device=DEVICE, dtype=data_type)
    input_3_gpu = torch.rand(broadcast_shape, device=DEVICE, dtype=data_type)
    output_gpu = torch.empty_like(input_1_gpu)

    input_1_gpu_addr = input_1_gpu.data_ptr()
    input_2_gpu_addr = input_2_gpu.data_ptr()
    input_3_gpu_addr = input_3_gpu.data_ptr()
    output_gpu_addr = output_gpu.data_ptr()

    context.set_tensor_address("input_1", int(input_1_gpu_addr))
    context.set_tensor_address("input_2", int(input_2_gpu_addr))
    context.set_tensor_address("input_3", int(input_3_gpu_addr))
    context.set_tensor_address("output", int(output_gpu_addr))
    cuda_stream = torch.cuda.Stream(device=DEVICE)
    enqueue_status = context.execute_async_v3(cuda_stream.cuda_stream)
    print(enqueue_status)
    assert enqueue_status
    cuda_stream.synchronize()
    output_data = output_gpu.cpu().numpy()

    golden_data = torch.relu(input_1_gpu + input_2_gpu - input_3_gpu)
    print(
        f"The maximum difference of result between torch and triton is "
        f"{torch.max(torch.abs(output_gpu - golden_data))}"
    )
    assert torch.allclose(output_gpu, golden_data, rtol=1e-03, atol=1e-05)

# unittest()