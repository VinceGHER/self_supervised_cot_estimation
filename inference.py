import numpy as np
import os
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import time
import yaml
import wandb
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from src.ml_orchestrator.transforms.transforms_builder import TransformBuilder
from src.models.model_builder import model_builder
from src.tools import check_file_path
import torch
TRT_LOGGER = trt.Logger()


class InferencerManager:
    def __init__(self,config_path,engine_file_path):
        self.config = yaml.safe_load(open(check_file_path(config_path)))
        self.transform_builder = TransformBuilder(self.config['transforms'])
        self.transform_common = self.transform_builder.build_transform_common_validation()
        self.transform_inputs = self.transform_builder.build_transforms_inputs_validation()


        self.engine = self.load_engine(engine_file_path)
        self.context = self.engine.create_execution_context() 
        self.stream = cuda.Stream()
        print("Model loaded successfully")

    def load_engine(self,engine_file_path):
        assert os.path.exists(engine_file_path)
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def run_inference(self, img_rgb, img_depth):
        start = time.time()
        engine = self.engine
        context = self.context
        stream = self.stream
        # Set input shape based on image dimensions for inference
        # context.set_binding_shape(engine.get_binding_index("input_rgb"), img_rgb.shape)
        # context.set_binding_shape(engine.get_binding_index("input_depth"), img_depth.shape)


        end = time.time()
        print("Time to set context: ", end - start)

        # Allocate host and device buffers
        bindings = []
        
        start = time.time()
        for binding in engine:
            binding_idx = engine.get_binding_index(binding)
            size = trt.volume(context.get_binding_shape(binding_idx))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            if engine.binding_is_input(binding):
                if str(binding) == "input_rgb":
                    input_buffer = np.ascontiguousarray(img_rgb)
                elif str(binding) == 'input_depth':
                    input_buffer = np.ascontiguousarray(img_depth)
                input_memory = cuda.mem_alloc(input_buffer.nbytes)
                bindings.append(int(input_memory))
                cuda.memcpy_htod_async(input_memory, input_buffer, stream)
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                bindings.append(int(output_memory))
        end = time.time()
        print("Time to allocate buffers: ", end - start)
        

        # Transfer input data to the GPU.

        # Run inference
        start = time.time()
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        stream.synchronize()
        end = time.time()
        print("Time to execute: ", end - start)
        # Transfer prediction output from the GPU.
        start  = time.time()
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        stream.synchronize()
        end = time.time()
        print("Time to copy output: ", end - start)
        # Synchronize the stream
        stream.synchronize()
        output_buffer = output_buffer.reshape(engine.get_binding_shape("output"))
        return output_buffer

    def predict(self, color_image, depth_image):
       
        # undistort the image
        


        # depth = torch.from_numpy(depth_numpy).float()
        # depth = depth.unsqueeze(0)
        # image = self.transform_inputs(color_image)
        # combined = torch.cat((image, depth), dim=0)
        # combined = self.transform_common(combined)

        # image = combined[:3, :, :]
        # depth = combined[3, :, :].unsqueeze(0)
        # image = image.permute(0, 1, 2)
        # depth = depth.permute(0, 1, 2)
       
        # Dummy model forward
        # Add dimension for batch
        # color_image_batch = image.unsqueeze(0)
        # depth_image_batch = depth.unsqueeze(0)
        # print(color_image_batch)
        # print(depth_image_batch)
        start_time = time.time()
        output = self.run_inference(color_image,depth_image)
                
        end_time = time.time()
        inference_time = end_time - start_time
        print("Inference time: ", inference_time)
        # print(output.shape)
        
                # Apply the colormap nipy_spectral
        cmap = plt.get_cmap('nipy_spectral')
        outnormalized = (output - 0) / (8-0)
        colored_array = cmap(outnormalized)  # This returns a 480x640x4 array with RGBA values

        # Convert the RGBA values to RGB (ignore the alpha channel)
        output_color = (colored_array[:, :, :3] * 255).astype(np.uint8)  # Convert to uint8 type

        return output,output_color,inference_time

def undistort_image(image, K, D):
    # Image to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Get the camera matrix
    K = np.array(K).reshape(3, 3)
    D = np.array(D)
    # Get the new camera matrix
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (image.shape[1], image.shape[0]), 0)
    # Undistort the image
    undistorted_image = cv2.undistort(image, K, D, None, new_K)
    # Image to RGB
    undistorted_image = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB)
    return undistorted_image
import queue 
import threading 
import time 

thread_exit_Flag = 0
class sample_Thread (threading.Thread): 
   def __init__(self, config,threadID, name, q,qo,queueLock): 
      threading.Thread.__init__(self) 
      self.threadID = threadID 
      self.name = name 
      self.q = q 
      self.qo = qo
      self.queueLock = queueLock
      self.config = config
   def run(self): 
      print ("initializing " + self.name) 
      process_data(self.config,self.name, self.q,self.qo,self.queueLock) 
      print ("Exiting " + self.name) 
class Preprocessor:
    def __init__(self,config_path,num_threads=4):
        self.queue = queue.Queue(1000)
        self.output_queue = queue.Queue(1000)
        self.queueLock = threading.Lock()
        self.threads=[]
        self.thread_exit_Flag = False
        self.num_threads = num_threads
        self.config = yaml.safe_load(open(check_file_path(config_path)))

    def start_threads(self):
        for i in range(self.num_threads):
            thread = sample_Thread(self.config,i, "Thread-"+str(i), self.queue,self.output_queue,self.queueLock)
            thread.start()
            self.threads.append(thread)
    def stop_threads(self):
        for t in self.threads:
            t.join()
        print("All threads stopped")

    def preprocess(self,image,depth):
        self.queue.put((image,depth))
def preprocess_data(config,color_image,depth_image):
    color_image = undistort_image(color_image, K, D)
    config = config
    d = np.nan_to_num(depth_image, nan=0)
    print(np.max(d),np.mean(d))
    depth = np.array(depth_image)/config['depth']['depth_to_meters']
    
    depth = np.nan_to_num(depth, nan=config['depth']['max_depth'])

    
    depth[depth > config['depth']['max_depth']] = config['depth']['max_depth']
    depth[depth <= config['depth']['min_depth']] = config['depth']['max_depth']
    depth_numpy = (depth - config['depth']['min_depth']) / (config['depth']['max_depth'] - config['depth']['min_depth'])
    print(np.max(depth_numpy),np.min(depth_numpy))
# cmap = plt.get_cmap('nipy_spectral')
# colored_array = cmap(depth_numpy)  # This returns a 480x640x4 array with RGBA values

    # Convert the RGBA values to RGB (ignore the alpha channel)
    # output_color = (colored_array[:, :, :3] * 255).astype(np.uint8)  # Convert to uint8 type
    # output_color =cv2.cvtColor(output_color, cv2.COLOR_RGB2BGR)
    # cv2.imshow('depth',output_color)
    # cv2.waitKey(0)

    mean= config['transforms']['normalize_input']['mean']
    std = config['transforms']['normalize_input']['std']

    color_image = color_image / 255.0
    color_image = (color_image - mean) / std

    image = color_image.transpose(2, 0, 1)
    color_image_batch = torch.from_numpy(np.expand_dims(image, axis=0).astype('float32'))
    depth = np.expand_dims(depth_numpy, axis=0)
    depth_image_batch = torch.from_numpy(np.expand_dims(depth, axis=0).astype('float32'))
    return color_image_batch,depth_image_batch
def process_data(config,threadName, q,qo,queueLock): 
    while not thread_exit_Flag: 
        queueLock.acquire() 
        if not q.empty(): 
            data = q.get() 
            queueLock.release() 
            color_image = data[0]
            depth_image = data[1]
            color_image_batch,depth_image_batch= preprocess_data(config,color_image,depth_image)
            queueLock.acquire() 
            qo.put((color_image_batch,depth_image_batch))
            queueLock.release() 
            print ("% s processing " % (threadName)) 
        else: 
            queueLock.release() 
            time.sleep(1) 


if __name__ == "__main__":
    print("Running TensorRT inference")
    K = np.array([3.7769427490234375e+02, 0., 3.2287564086914062e+02, 0., 0.,
        3.7732968139648438e+02, 2.4526431274414062e+02, 0., 0.]).reshape(3,3)
    D =  np.array([ 0., 0., 0., 0., 0. ])
    img_rbg = np.array(Image.open("test_rgb.jpg"))
    img_depth = np.array(Image.open("test_depth.png"))
    inferencerManager = InferencerManager("artifacts/saved_model:v30/config.yaml","asymformer.engine")
    preprocessor = Preprocessor("artifacts/saved_model:v30/config.yaml")
    preprocessor.start_threads()
    total_inference_time = 0
    start = time.time()
    count=0
    for _ in range(1000):
        preprocessor.preprocess(img_rbg,img_depth)
    while count<1000:
        preprocessor.queueLock.acquire() 
        if not preprocessor.output_queue.empty():
            output = preprocessor.output_queue.get()
            preprocessor.queueLock.release() 
            inferencerManager.predict(output[0],output[1])
            count+=1
            print("Count",count)
        else:
            preprocessor.queueLock.release()
            time.sleep(1)
    end = time.time()
    thread_exit_Flag = True
    preprocessor.stop_threads()
    avg_inference_time = (end-start) / 1000 * 1000
    print(f"Average inference time over 100 rounds: {avg_inference_time:.4f} ms")
  