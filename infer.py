import numpy as np
import os
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import time
import matplotlib.pyplot as plt
import cProfile
import pstats
import PIL.Image
TRT_LOGGER = trt.Logger()

class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print(f"Reading engine from file {engine_file_path}")
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def create_context(engine, batch_size):
    print("Creating context")
    bindings = []
    inputs = []
    outputs = []

    for binding in engine:
        binding_idx = engine.get_binding_index(binding)
        shape = (batch_size, *engine.get_binding_shape(binding_idx)[1:])
        if engine.binding_is_input(binding):
            context.set_binding_shape(binding_idx, shape)
        print(f"Binding {binding}: {shape}")
        size = trt.volume(shape)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return bindings, inputs, outputs

def memory_h(inputs, stream, img_rgb, img_depth, sync=False):
    np.copyto(inputs[0].host,img_rgb.ravel())
    np.copyto(inputs[1].host,img_depth.ravel())
    for input_buffer in inputs:
        cuda.memcpy_htod_async(input_buffer.device, input_buffer.host, stream)
    if sync:
        stream.synchronize()
def memory_d(outputs, stream, sync=False):
    for output_buffer in outputs:
        cuda.memcpy_dtoh_async(output_buffer.host, output_buffer.device, stream)
    if sync:
        stream.synchronize()
def run_one_inference(context, bindings, stream, sync=False):
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    if sync:
        stream.synchronize()

def infer(engine, context, img_rgb, img_depth, bindings, inputs, outputs,stream):

    memory_h(inputs, stream, img_rgb, img_depth)
    run_one_inference(context, bindings, stream)
    memory_d(outputs, stream)

    return outputs

print("Running TensorRT inference")

engine_file = "models/asymformer-v19-4.engine"
batch_size = 4

def f8_alt(x):
    return "%8.3f" % (x * 1000)

pstats.f8 = f8_alt


RGB = "datasets/dataset-2024-05-29_15-53-37/test/rgb/1712269354.732150.jpg"
DEPTH = "datasets/dataset-2024-05-29_15-53-37/test/depth/1712269354.732150.png"

def run_profile(engine, context, img_rgb, img_depth, bindings, inputs, outputs,stream):
    start = time.time()
    memory_h(inputs, stream, img_rgb, img_depth,True)
 
    run_one_inference(context, bindings, stream,True)
 
    memory_d(outputs, stream,True)

    end = time.time()
    return end - start
    
with load_engine(engine_file) as engine:
    img_rgb = np.random.rand(batch_size, 3, 480, 640).astype(np.float16)
    img_depth = np.random.rand(batch_size, 1, 480, 640).astype(np.float16)
    with engine.create_execution_context() as context:
        bindings, inputs, outputs = create_context(engine, batch_size)
        stream = cuda.Stream()
        start_time = time.time()
        for _ in range(5):  # warm-up
            infer(engine, context, img_rgb, img_depth, bindings, inputs, outputs,stream)
        inference_time = time.time() - start_time
        print(f"Run inference time: {inference_time * 1000:.2f} ms for")
        print("Warm-up done")


        # Profile the model
        print("Profiling the model")
        stream.synchronize()
        pr = cProfile.Profile()
        pr.enable()
        inference_time = run_profile(engine, context, img_rgb, img_depth, bindings, inputs, outputs,stream)
        pr.disable()        
        print(f"Run inference time: {(inference_time)*1000:.2f} ms")
        # print(f"Run inference time: {inference_time * 1000:.2f} ms per batch of image")
        pr.print_stats(sort="cumtime")
        print("Profiling done")


        # Test throughput
        
        index = 0
        count = 0
        
        # generate 100 random images
        img_rgb = np.random.rand(100, batch_size,3, 480, 640).astype(np.float16)
        img_depth = np.random.rand(100,batch_size, 1, 480, 640).astype(np.float16)

        # img_rgb = np.ones((100, batch_size,3, 480, 640)).astype(np.float16)
        # img_depth = np.ones((100,batch_size, 1, 480, 640)).astype(np.float16)
        max_time = 10
        start = time.time()
        while(time.time() - start < max_time):
            o=infer(engine, context, img_rgb[index], img_depth[index], bindings, inputs, outputs,stream)
            index +=1
            count+=batch_size
            if index > 99:
                index = 0
        
        print(f"Ran {count} inferences in {max_time} seconds")
        print("throughput: ",count / max_time, "fps") 


        # Test average inference time
        start = time.time()
        for i in range(100):
            ô=infer(engine, context, img_rgb[i], img_depth[i], bindings, inputs, outputs,stream)
        stream.synchronize()
        total_inference_time = time.time() - start 
        avg_inference_time = total_inference_time / (100 * batch_size) * 1000
        
        print(f"Average inference time over 100 rounds: {avg_inference_time:.4f} ms")

        o = infer(engine, context, img_rgb[0], img_depth[0], bindings, inputs, outputs, stream)
        o = np.reshape(o[0].host, (batch_size, 1, 480, 640))

        plt.imshow(o[0][0])
        plt.show()
