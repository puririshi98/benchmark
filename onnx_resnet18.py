import torch
import torch.cuda as cuda
import torchvision.models as models
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import time
import tensorrt as trt




def build_engine(model_path):
	with trt.Builder(TRT_LOGGER) as builder,builder.create_network(EXPLICIT_BATCH) as network,trt.OnnxParser(network, TRT_LOGGER) as parser, builder.create_builder_config() as config:
		config.max_workspace_size = 1<<30
		# config.max_batch_size = 1
		with open(model_path, "rb") as f:
			parser.parse(f.read())
		engine = builder.build_engine(network, config)
	return engine
# def inference(engine, context, inputs, out_cpu, in_gpu, out_gpu, stream):
# async version
# with engine.create_execution_context() as context:  # cost time to initialize
# cuda.memcpy_htod_async(in_gpu, inputs, stream)
# context.execute_async(1, [int(in_gpu), int(out_gpu)], stream.handle, None)
# cuda.memcpy_dtoh_async(out_cpu, out_gpu, stream)
# stream.synchronize()
def inference(engine, context, inputs, h_input, h_output, d_input, d_output, stream):
	cuda.memcpy_htod_async(d_input, h_input, stream)
	# Run inference.
	context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
	# Transfer predictions back from the GPU.
	cuda.memcpy_dtoh_async(h_output, d_output, stream)
	return h_output

	'''
	# sync version
	cuda.memcpy_htod(in_gpu, inputs,stream)
	context.execute(1, [int(in_gpu), int(out_gpu)])
	cuda.memcpy_dtoh(out_cpu, out_gpu,stream)
	return out_cpu'''


def alloc_buf(engine,dtype):
	h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=dtype)
	h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=dtype)
	# Allocate device memory for inputs and outputs.
	d_input = cuda.mem_alloc(h_input.nbytes)
	d_output = cuda.mem_alloc(h_output.nbytes)
	stream = cuda.Stream()

	return h_input, h_output, d_input, d_output, stream


if __name__ == "__main__":
	dummy_input = torch.randn(1, 3, 224, 224).cuda()
	model = models.resnet18().cuda()
	torch.onnx.export(model, dummy_input, "res18.onnx", verbose=False)

	TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
	model_path ="res18.onnx"
	input_size = 224
	TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
	EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
	engine = build_engine(model_path)
	print("Engine Created :", type(engine))
	context = engine.create_execution_context()
	print("Context executed ", type(context))
	time_sum=0
	for i in range(5):
		inputs = np.random.random((1, 3, input_size, input_size)).astype(np.float32)
		t1 = time.time()
		# in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(engine)
		h_input, h_output, d_input, d_output, stream = alloc_buf(engine, np.float32)
		res = inference(engine, context, inputs.reshape(-1), h_input, h_output, d_input, d_output, stream)
		# print(type(res))
		
		time_sum+=time.time()-t1
	print("using fp32 mode:")
	print("avg cost time: ", round(1000.0*time_sum/5.0,4),'ms')
	time_sum=0
	for i in range(5):
		inputs = np.random.random((1, 3, input_size, input_size)).astype(np.float16)
		t1 = time.time()
		# in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(engine)
		h_input, h_output, d_input, d_output, stream = alloc_buf(engine,np.float16)
		res = inference(engine, context, inputs.reshape(-1), h_input, h_output, d_input, d_output, stream)
		# print(type(res))
		
		time_sum+=time.time()-t1
	print("using fp16 mode:")
	print("avg cost time: ", round(1000*time_sum/(i+1),4),'ms')
	time_sum=0
	model1=model.eval()
	for i in range(5):
		inputs = torch.tensor(np.random.random((1, 3, input_size, input_size)).astype(np.float32)).cuda()
		t1 = time.time()
		# in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(engine)
		out=model1(inputs)
		# print(type(res))
		
		time_sum+=time.time()-t1
	print("using fp32 mode:")
	print("avg cost time: ", round(1000.0*time_sum/5.0,4),'ms')
	time_sum=0
	halfmodel=model.half().eval()
	for i in range(5):
		inputs = torch.tensor(np.random.random((1, 3, input_size, input_size)).astype(np.float16)).cuda().half()
		
		t1 = time.time()
		# in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(engine)
		out=halfmodel(inputs)
		# print(type(res))
		
		time_sum+=time.time()-t1
	print("using fp16 mode:")
	print("avg cost time: ", round(1000.0*time_sum/5.0,4),'ms')

# self.model.eval()
# 			torch.backends.cudnn.benchmark = bench
# 			with torch.no_grad():
# 				if precision == 'fp16':
# 					self.model = self.model.half()
# 				if graphs:
# 					s = torch.cuda.Stream()
# 					torch.cuda.synchronize()
# 					with torch.cuda.stream(s):
# 						nvtx.range_push('warming up')
# 						print('warming up')
# 						for _ in range(5):
# 							self._step_eval(precision)
# 						nvtx.range_pop()
# 						torch.cuda.empty_cache()
# 						g = torch.cuda._Graph()
# 						torch.cuda.synchronize()
# 						nvtx.range_push('capturing graph')
# 						print('capturing graph')
# 						g.capture_begin()
# 						self._step_eval(precision)
# 						g.capture_end()
# 						nvtx.range_pop()
# 						torch.cuda.synchronize()
# 					nvtx.range_push('replaying')
# 					print('replaying')
# 					for _ in range(100):
# 						g.replay()
# 						torch.cuda.synchronize()
# 					nvtx.range_pop()
				