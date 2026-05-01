import argparse
import threading
import time
import numpy as np
import tensorrt as trt
import cuda.cuda as cuda
import cuda.cudart as cudart
from .data_generate import generate_input_buffers, generate_output_buffers


def cuda_call(call):
    def _cudaGetErrorEnum(error):
        if isinstance(error, cuda.CUresult):
            err, name = cuda.cuGetErrorName(error)
            return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
        elif isinstance(error, cudart.cudaError_t):
            return cudart.cudaGetErrorName(error)[1]
        else:
            raise RuntimeError("Unknown error type: {}".format(error))

    err, res = call[0], call[1:]
    if err.value:
        raise RuntimeError(
            "CUDA error code={}({})".format(
                err.value, _cudaGetErrorEnum(err)
            )
        )
    if len(res) == 1:
        return res[0]
    elif len(res) == 0:
        return None
    else:
        return res


class TRTInferenceEngine:
    def __init__(self, serialized_engine, gpu_id, trt_log):
        self.gpu_id = gpu_id

        cuda_call(cudart.cudaSetDevice(gpu_id))
        cuda_call(cudart.cudaFree(0))

        self.runtime = trt.Runtime(trt_log)
        self.engine = self.runtime.deserialize_cuda_engine(serialized_engine)
        self.context = self.engine.create_execution_context()

        self.input_bindings = []
        self.output_bindings = []
        self.stream = cuda_call(cudart.cudaStreamCreate())
        self.bsz = None
        self.input_buffers = {}
        self.output_buffers = {}

    def allocate_buffers(self, custom_buffers):
        self.input_buffers = generate_input_buffers(self.input_bindings, custom_buffers)
        self.output_buffers = generate_output_buffers(self.output_bindings)

        for input_binding in self.input_bindings:
            binding_name = input_binding["name"]
            cuda_call(cudart.cudaMemcpyAsync(
                input_binding["dptr"],
                self.input_buffers[binding_name],
                input_binding["nbytes"],
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream
            ))
            self.context.set_tensor_address(
                binding_name,
                input_binding["dptr"]
            )

        for output_binding in self.output_bindings:
            self.context.set_tensor_address(
                output_binding["name"],
                output_binding["dptr"]
            )

    def set_input_shapes(self, inference_input_shapes):
        # set input shapes
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            if self.engine.binding_is_input(i):
                engine_dims = self.engine.get_binding_shape(i)
                is_dynamic_shape = False
                for d in list(engine_dims):
                    if d < 0:
                        is_dynamic_shape = True
                if is_dynamic_shape:
                    profile_shape = self.engine.get_profile_shape(0, i)
                    if name not in inference_input_shapes:
                        print(
                            "Dynamic dimensions required for input: {}, but no shapes were provided. Automatically overriding shape to: {}".format(
                                name, "x".join([str(i) for i in profile_shape[0]])
                            )
                        )
                        shape = profile_shape[0]
                    else:
                        shape = inference_input_shapes[name]
                    self.context.set_input_shape(name, shape)
                else:
                    if name in inference_input_shapes:
                        print(
                            "Static dimensions for input: {}, no need to provide shapes. The --shape parameter will be ignored.".format(
                                name
                            )
                        )
                shape = self.context.get_binding_shape(i)
                print(
                    "Input inference shape: {}={}".format(
                        name, "x".join([str(i) for i in shape])
                    )
                )

        # After all input shapes set, output shape can be acquired
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.context.get_binding_shape(i)
            if 0 in shape:
                print(f"Warning: Binding '{name}' has a shape with zero element: {shape}. Skipping memory allocation.")
                continue
            if self.engine.binding_is_input(i):
                if self.bsz is None:
                    self.bsz = shape[0]
                size = np.dtype(trt.nptype(dtype)).itemsize
                for s in shape:
                    size *= s
            else:
                size = self.context.get_max_output_size(name)
            err, dptr = cudart.cudaMalloc(size)
            assert err == cudart.cudaError_t.cudaSuccess
            binding_data = {
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "dptr": dptr,
                "nbytes": size,
            }
            if self.engine.binding_is_input(i):
                self.input_bindings.append(binding_data)
            else:
                self.output_bindings.append(binding_data)

    def get_output_buffer(self):
        for output_binding in self.output_bindings:
            binding_name = output_binding["name"]
            cuda_call(cudart.cudaMemcpyAsync(
                self.input_buffers[binding_name],
                output_binding["dptr"],
                input_binding["nbytes"],
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream
            ))

        cuda_call(cudart.cudaStreamSynchronize(self.stream))
        return self.output_buffers

    def infer(self):
        self.context.execute_async_v3(stream_handle=self.stream)


    def cleanup(self):
        cuda_call(cudart.cudaStreamDestroy(self.stream))

        for inp_binding in self.input_bindings:
            cuda_call(cudart.cudaFree(inp_binding["dptr"]))
        for out_binding in self.output_bindings:
            cuda_call(cudart.cudaFree(out_binding["dptr"]))

        del self.context
        del self.engine
        del self.runtime


def inference_worker(trt_log, serialized_engine, gpu_id, warm_up, iterations, results_dict, thread_id, custom_buffers, input_shapes=None):
    print(f"[Thread {thread_id}] Starting on GPU {gpu_id}")

    try:
        engine = TRTInferenceEngine(serialized_engine, gpu_id, trt_log)

        engine.set_input_shapes(input_shapes)
        engine.allocate_buffers(custom_buffers)

        times = []

        for i in range(warm_up):
            engine.infer()

        print(f"[Thread {thread_id}] Running {iterations} inference iterations on GPU {gpu_id}...")

        cuda_call(cudart.cudaStreamSynchronize(engine.stream))
        start_time = time.perf_counter()
        for i in range(iterations):
            engine.infer()
        cuda_call(cudart.cudaStreamSynchronize(engine.stream))
        end_time = time.perf_counter()
        fps = iterations * engine.bsz / (end_time - start_time)
        throughput = iterations / (end_time - start_time)

        results_dict[thread_id] = {
            "gpu_id": gpu_id,
            "iterations": iterations,
            "fps": fps,
            "throughput": throughput,
            "bsz": engine.bsz,
        }

        print(f"[Thread {thread_id}] Completed on GPU {gpu_id}")

        engine.cleanup()

    except Exception as e:
        print(f"[Thread {thread_id}] Error on GPU {gpu_id}: {e}")
        results_dict[thread_id] = {"error": str(e)}
