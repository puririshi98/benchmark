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

import timm.models.vision_transformer
import timm
if __name__ == "__main__":
	dummy_input = torch.randn(1, 3, 224, 224).cuda()
	model = timm.create_model('vit_small_patch16_224', pretrained=False, scriptable=True).cuda()

	torch.onnx.export(model, dummy_input, "vt.onnx", verbose=False)

	TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
	model_path ="vt.onnx"
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
	print("using onnxTRT fp32 mode:")
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
	print("using onnxTRT fp16 mode:")
	print("avg cost time: ", round(1000*time_sum/(i+1),4),'ms')
	time_sum=0
	model1=model.float().eval()
	for i in range(5):
		inputs = torch.tensor(np.random.random((1, 3, input_size, input_size))).cuda().float()
		t1 = time.time()
		# in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(engine)
		out=model1(inputs)
		# print(type(res))
		
		time_sum+=time.time()-t1
	print("using torch fp32 mode:")
	print("avg cost time: ", round(1000.0*time_sum/5.0,4),'ms')
	time_sum=0
	halfmodel=model.half().eval()
	for i in range(5):
		inputs = torch.tensor(np.random.random((1, 3, input_size, input_size))).cuda().half()
		
		t1 = time.time()
		# in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(engine)
		out=halfmodel(inputs)
		# print(type(res))
		
		time_sum+=time.time()-t1
	print("using torch fp16 mode:")
	print("avg cost time: ", round(1000.0*time_sum/5.0,4),'ms')
	time_sum=0
	torch.backends.cudnn.benchmark = True
	model1=model.float().eval()
	s = torch.cuda.Stream()
	torch.cuda.synchronize()
	inputs = torch.tensor(np.random.random((1, 3, input_size, input_size))).cuda().float()
	with torch.cuda.stream(s):
		for _ in range(5):
			out=model1(inputs)
		torch.cuda.empty_cache()
		g = torch.cuda._Graph()
		torch.cuda.synchronize()
		g.capture_begin()
		out=model1(inputs)
		g.capture_end()
		torch.cuda.synchronize()

	for _ in range(5):
		t1=time.time()
		g.replay()
		torch.cuda.synchronize()
		
		time_sum+=time.time()-t1
	print("using cudagraphsfp32 mode:")
	print("avg cost time: ", round(1000.0*time_sum/5.0,4),'ms')
	time_sum=0
	torch.backends.cudnn.benchmark = True
	model2=model.half().eval()
	s = torch.cuda.Stream()
	torch.cuda.synchronize()
	inputs = torch.tensor(np.random.random((1, 3, input_size, input_size))).cuda().half()
	with torch.cuda.stream(s):
		for _ in range(5):
			out=model2(inputs)
		torch.cuda.empty_cache()
		g = torch.cuda._Graph()
		torch.cuda.synchronize()
		g.capture_begin()
		out=model2(inputs)
		g.capture_end()
		torch.cuda.synchronize()

	for _ in range(5):
		t1=time.time()
		g.replay()
		torch.cuda.synchronize()
		
		time_sum+=time.time()-t1
	print("using cudagraphsfp16 mode:")
	print("avg cost time: ", round(1000.0*time_sum/5.0,4),'ms')
	torch._C._jit_set_nvfuser_enabled(True)
	torch._C._jit_set_texpr_fuser_enabled(False)
	torch._C._jit_set_profiling_executor(True)
	torch._C._jit_set_profiling_mode(True)
	torch._C._jit_override_can_fuse_on_cpu(False)
	torch._C._jit_override_can_fuse_on_gpu(False)
	torch._C._jit_set_bailout_depth(20)
	time_sum=0
	torch.backends.cudnn.benchmark = True
	model1=torch.jit.script(models.resnet18().cuda()).float().eval()
	s = torch.cuda.Stream()
	torch.cuda.synchronize()
	inputs = torch.tensor(np.random.random((1, 3, input_size, input_size))).cuda().float()
	with torch.cuda.stream(s):
		for _ in range(5):
			out=model1(inputs)
		torch.cuda.empty_cache()
		g = torch.cuda._Graph()
		torch.cuda.synchronize()
		g.capture_begin()
		out=model1(inputs)
		g.capture_end()
		torch.cuda.synchronize()

	for _ in range(5):
		t1=time.time()
		g.replay()
		torch.cuda.synchronize()
		
		time_sum+=time.time()-t1
	print("using nvfusedcudagraphsfp32 mode:")
	print("avg cost time: ", round(1000.0*time_sum/5.0,4),'ms')
	time_sum=0
	torch.backends.cudnn.benchmark = True
	model2=torch.jit.script(models.resnet18().cuda()).half().eval()
	s = torch.cuda.Stream()
	torch.cuda.synchronize()
	inputs = torch.tensor(np.random.random((1, 3, input_size, input_size))).cuda().half()
	with torch.cuda.stream(s):
		for _ in range(5):
			out=model2(inputs)
		torch.cuda.empty_cache()
		g = torch.cuda._Graph()
		torch.cuda.synchronize()
		g.capture_begin()
		out=model2(inputs)
		g.capture_end()
		torch.cuda.synchronize()

	for _ in range(5):
		t1=time.time()
		g.replay()
		torch.cuda.synchronize()
		
		time_sum+=time.time()-t1
	print("using nvfusedcudagraphsfp16 mode:")
	print("avg cost time: ", round(1000.0*time_sum/5.0,4),'ms')