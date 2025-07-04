import ctypes
from typing import List
import numpy as np
import tensorrt as trt
import torch
from collections import OrderedDict
import onnx
import onnx_graphsurgeon as gs
import numpy as np
import triton
import triton.language as tl
from cuda import cudart

def datatype_np_to_torch(datatype_np):
    if datatype_np == np.float32:
        return torch.float32
    if datatype_np == np.float16:
        return torch.float16
    if datatype_np == np.int8:
        return torch.int8
    if datatype_np == np.int32:
        return torch.int32
    if datatype_np == bool:
        return torch.bool
    if datatype_np == np.uint8:
        return torch.uint8
    if datatype_np == np.int64:
        return torch.int64
    return None

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(0)  # tl.program_id(0) -> blockIdx.x
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)

# Let's also declare a helper function to (1) allocate the `c` tensor
# and (2) enqueue the above kernel with appropriate grid/block sizes:
def add_layer(a: torch.Tensor, b: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(a)
    assert a.is_cuda and b.is_cuda and output.is_cuda
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](a, b, output, n_elements, BLOCK_SIZE=256)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output

class AddPlugin(trt.IPluginV2DynamicExt):
    def __init__(self: trt.IPluginV2DynamicExt): 
        trt.IPluginV2DynamicExt.__init__(self)
        self.plugin_name = "AddPlugin"  # necessary as function `getPluginType` in C++
        self.plugin_version = "1"  # necessary as function `getPluginVersion` in C++
        self.num_outputs = 1  # necessary as function `getNbOutputs` in C++
        self.plugin_namespace = ""  # necessary as function `setPluginNamespace`/ `getPluginNamespace` in C++
        self.device = 0  # default device is cuda:0, can be get by `cuda.cuDeviceGet(0)`
        self.plugin_type = self.plugin_name
        return

    def clone(self) -> trt.IPluginV2DynamicExt:
        cloned_plugin = AddPlugin()
        cloned_plugin.__dict__.update(self.__dict__)
        return cloned_plugin
    def get_workspace_size(self, dptd_in: List[trt.PluginTensorDesc], dptd_out: List[trt.PluginTensorDesc]) -> int:
        """ Return necessary workspace to this plugin """
        return 0
    def get_output_dimensions(self, output_index: int, inputs: List[trt.DimsExprs], expr_builder: trt.IExprBuilder) -> trt.DimsExprs:
        output_dims = trt.DimsExprs(inputs[0])
        return output_dims

    def get_output_datatype(self: trt.IPluginV2DynamicExt, index: int, input_types: List[trt.DataType]) -> trt.DataType:
        """ Describe the output data type based-on inputs

        In this plugin the output data type is the same than input because 
        add op no change the data type      
        """        
        return input_types[0]

    def get_output_shapes(self, inputs: List[trt.DimsExprs], shape_inputs: List[trt.DimsExprs], expr_builder: trt.IExprBuilder) -> List[trt.DimsExprs]:
        """ Describe the output shape based-on inputs

        In this plugin the output is the same than input because 
        add op no change the shape          
        """
        output_dims = trt.DimsExprs(inputs[0])
        return [output_dims]

    def supports_format_combination(self, pos: int, in_out: List[trt.PluginTensorDesc], num_inputs: int) -> bool:
        res = False
        desc = in_out[pos]
        if pos in [0, 1]:
            # inputs should be float16 or float32
            res = (desc.type == trt.float32 or desc.type == trt.float16) and desc.format == trt.TensorFormat.LINEAR
        # check output
        elif pos == 2:
            # output should have the same type as the input
            res = desc.type == in_out[0].type and desc.format == trt.TensorFormat.LINEAR
        return res

    def configure_plugin(self, inp, out):
        return

    def get_serialization_size(self):
        return 0
    


    def enqueue(self, input_desc: List[trt.PluginTensorDesc], output_desc: List[trt.PluginTensorDesc], inputs: List[int], outputs: List[int], workspace: int, stream: int) -> None:
     """ Execute kernel on GPU

     Implement C = A + B using OpenAI Triton
     
     Args:
         input_desc: input tensor description list
         output_desc: output tensor description list
         inputs: pointer list to inputs
         outputs: pointer list to outputs
         workspace
         stream                
     """
     A_DATA_TYPE = trt.nptype(input_desc[0].type)
     B_DATA_TYPE = trt.nptype(input_desc[1].type)     
     assert A_DATA_TYPE == B_DATA_TYPE

     A_SHAPE = tuple(input_desc[0].dims)
     B_SHAPE = tuple(input_desc[1].dims)
     assert A_SHAPE == B_SHAPE

     A_N_ELEMENT = np.prod(A_SHAPE)
     B_N_ELEMENT = np.prod(B_SHAPE)

     # pointer -> numpy.ndarray -> torch.Tensor
     BUFFER_SIZE = A_N_ELEMENT * np.dtype(A_DATA_TYPE).itemsize   
     cpp_data_type = ctypes.c_int16 if A_DATA_TYPE == np.float16 else ctypes.c_float

     A_POINTER = ctypes.cast(inputs[0], ctypes.POINTER(cpp_data_type * A_N_ELEMENT))[0]
     A_NP_ARRAY = np.ndarray(A_SHAPE, dtype=A_DATA_TYPE, buffer=A_POINTER)
     A_TENSOR = torch.as_tensor(A_NP_ARRAY, device="cuda")

     B_POINTER = ctypes.cast(inputs[1], ctypes.POINTER(cpp_data_type * B_N_ELEMENT))[0]
     B_NP_ARRAY = np.ndarray(B_SHAPE, dtype=B_DATA_TYPE, buffer=B_POINTER)
     B_TENSOR = torch.as_tensor(B_NP_ARRAY, device="cuda")

     # C = A + B with triton add layer
     C_TENSOR = add_layer(A_TENSOR, B_TENSOR)
     
     # copy add output to buffer
     kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
     cudart.cudaMemcpyAsync(outputs[0], C_TENSOR.data_ptr(), BUFFER_SIZE, kind, stream)
     return


class AddPluginCreator(trt.IPluginCreator):

    def __init__(self):
        trt.IPluginCreator.__init__(self)
        self.name = "AddPlugin"  # necessary as function `getPluginName` in C++
        self.plugin_version = "1"  # necessary as function `getPluginVersion` in C++
        self.plugin_namespace = ""  # necessary as function `setPluginNamespace`/ `getPluginNamespace` in C++
        self.field_names = trt.PluginFieldCollection([])
        return

    def create_plugin(self, name: str, field_collection: trt.PluginFieldCollection):
        return AddPlugin()
    
    def deserialize_plugin(self, name: str, serialized_plugin):
        return AddPlugin()

if __name__ == "__main__":
    onnx_path = "add_model.onnx"
    # create input variables with batch fixed = 1
    input_A = gs.Variable(name="A", dtype=np.float32, shape=(1, "N"))
    input_B = gs.Variable(name="B", dtype=np.float32, shape=(1, "N"))

    # create output variable 
    output = gs.Variable(name="C", dtype=np.float32)

    # create node with custom plugin op
    add_plugin_node = gs.Node(
        name="AddPlugin",
        op="AddPlugin", 
        inputs=[input_A, input_B], 
        outputs=[output],
        #attrs={"pads": pads},
    )

    graph = gs.Graph(
      nodes=[add_plugin_node], 
      inputs=[input_A, input_B], 
      outputs=[output], 
      opset=17
    )
    onnx.save(gs.export_onnx(graph), onnx_path)
    # build engine
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) 

    # register plugin creator
    plg_registry = trt.get_plugin_registry()
    my_plugin_creator = AddPluginCreator()
    plg_registry.register_creator(my_plugin_creator, "")

    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print(f"ERROR: Failed to parse the ONNX file {onnx_path}")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    # set opt profile
    profile = builder.create_optimization_profile()
    # define range
    min_shape = [1] + [1]
    opt_shape = [1] + [1000]
    max_shape = [1] + [10000]
    for input in inputs:
        profile.set_shape(input.name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)    

    engine_bytes = builder.build_serialized_network(network, config) 
    engine_path = "add_model.engine"
    with open(engine_path, "wb") as f:    
        f.write(engine_bytes)
    
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as plan:
        engine = runtime.deserialize_cuda_engine(plan.read())

    # define inputs
    A = np.ones((1, 3))
    B = np.ones((1, 3))

    # create stream
    err, stream = cudart.cudaStreamCreate()

    # create Execution Context from the engine (analogy to a GPU context, or a CPU process)
    context = engine.create_execution_context()                                 

    # get i/o tensors names
    tensor_name_list = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]

    # set runtime size of input tensor if using Dynamic-Shape mode
    context.set_input_shape(tensor_name_list[0], A.shape)                      
    context.set_input_shape(tensor_name_list[1], B.shape)

    # Print information of input / output tensors
    for name in tensor_name_list:                                               
        mode = engine.get_tensor_mode(name)
        data_type = engine.get_tensor_dtype(name)
        buildtime_shape = engine.get_tensor_shape(name)
        runtime_shape = context.get_tensor_shape(name)
        print(f"{'Input ' if mode == trt.TensorIOMode.INPUT else 'Output'}->{data_type}, {buildtime_shape}, {runtime_shape}, {name}")

    device = "cuda:0"
    # prepare the memory buffer on host and device
    buffer = OrderedDict()                                                      
    for name in tensor_name_list:
        data_type = engine.get_tensor_dtype(name)
        runtime_shape = context.get_tensor_shape(name)
        buffer[name] = torch.empty(tuple(runtime_shape), dtype=datatype_np_to_torch(data_type), device=device)

    # model input
    data_A = torch.ones((1,3), dtype=torch.float32)   
    data_B = torch.ones((1,3), dtype=torch.float32)   

    # set runtime data, MUST use contiguous, it is a SERIOUS lesson
    buffer[tensor_name_list[0]] = data_A.contiguous().to(device)                 
    buffer[tensor_name_list[1]] = data_B.contiguous().to(device)     

    for name in tensor_name_list:
        # bind address of buffer to context
        context.set_tensor_address(name, buffer[name].data_ptr())               

    # do inference computation
    context.execute_async_v3(stream)                                                 

    # synchronize to get outputs
    cudart.cudaStreamSynchronize(stream)
    for name in tensor_name_list:
        print(name)
        print(buffer[name])
    if torch.all(buffer["C"] == 2):
        print("PASS")
    else:
        print("FAIL")
    
    del network
    del parser
    del context
    del engine
    del runtime
    del builder