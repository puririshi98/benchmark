import torch
import torch.cuda as cuda
import torchvision.models as models
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import time
import tensorrt as trt
import torch.cuda.nvtx as nvtx
import os
import timm.models.efficientnet
import timm



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
	h_input[:]=inputs.reshape(-1)
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
	h_input = cuda.pagelocked_empty(trt.volume((1,3,224,224)), dtype=dtype)
	h_output = cuda.pagelocked_empty(trt.volume((1,3,224,224)), dtype=dtype)
	# Allocate device memory for inputs and outputs.
	d_input = cuda.mem_alloc(h_input.nbytes)
	d_output = cuda.mem_alloc(h_output.nbytes)
	stream = cuda.Stream()

	return h_input, h_output, d_input, d_output, stream

if __name__ == "__main__":
	if not os.path.exists(os.getcwd()+os.sep+'ef.onnx'):
		dummy_input = torch.randn(1, 3, 224, 224).cuda()
		model = timm.create_model('mixnet_m', pretrained=False, scriptable=True).cuda().float().eval()

		torch.onnx.export(model, dummy_input, "ef.onnx", verbose=False)

	TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
	model_path ="ef.onnx"
	input_size = 224
	TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
	EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
	engine = build_engine(model_path)
	print("Engine Created :", type(engine))
	context = engine.create_execution_context()
	print("Context executed ", type(context))

	time_sum=0
	h_input, h_output, d_input, d_output, stream = alloc_buf(engine, np.float32)
	for i in range(5):
		inputs = np.random.random((1, 3, input_size, input_size)).astype(np.float32)
		t1 = time.time()
		nvtx.range_push("Singular Inference")
		res = inference(engine, context, inputs, h_input, h_output, d_input, d_output, stream)
		# print(type(res))
		nvtx.range_pop()
		time_sum+=time.time()-t1
		if i!=0:
			if (previous_out == h_output).all():
				print("inputs are changing but outputs are not")
		previous_out=np.copy(h_output.copy())
	print("using onnxTRT fp32 mode:")
	print("avg cost time: ", round(1000.0*time_sum/5.0,4),'ms')

	#fp16 cudagraphsonnxTRT
	time_sum=0
	torch.backends.cudnn.benchmark = True
	s = torch.cuda.Stream()
	torch.cuda.synchronize()
	h_input, h_output, d_input, d_output, stream = alloc_buf(engine, np.float16)
	oginputs = np.random.random((1, 3, input_size, input_size)).astype(np.float16)
	# ogoutputs = np.random.random((1, 3, input_size, input_size)).astype(np.float16)
	# oginputs = torch.randn((1, 3, input_size, input_size)).cuda().half()
	# ogoutputs = torch.randn((1, 3, input_size, input_size)).cuda().half()
	nvtx.range_push('warmup and capture')
	with torch.cuda.stream(s):
		for _ in range(5):
			inputs = np.random.random((1, 3, input_size, input_size)).astype(np.float16)
			# inputs = torch.randn((1, 3, input_size, input_size)).cuda().half()
			oginputs[:] = inputs
			# ogoutputs[:] = oginputs * 2
			# oginputs.copy_(inputs)
			# ogoutputs.copy_(oginputs * 2)
			h_input[:]=oginputs.reshape(-1)
			cuda.memcpy_htod_async(d_input, h_input, stream)
			# Run inference.
			context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
			# Transfer predictions back from the GPU.
			cuda.memcpy_dtoh_async(h_output, d_output, stream)
		
		torch.cuda.empty_cache()
		g = torch.cuda._Graph()
		torch.cuda.synchronize()
		inputs = np.random.random((1, 3, input_size, input_size)).astype(np.float16)
		# inputs = torch.randn((1, 3, input_size, input_size)).cuda().half()
		oginputs[:] = inputs
		# oginputs.copy_(inputs)
		h_input[:] = oginputs.reshape(-1)
		cuda.memcpy_htod_async(d_input, h_input, stream)
		g.capture_begin()
		context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
		# ogoutputs[:] = oginputs * 2
		# ogoutputs.copy_(oginputs * 2)
		g.capture_end()
		cuda.memcpy_dtoh_async(h_output, d_output, stream)
		torch.cuda.synchronize()
	nvtx.range_pop()
	nvtx.range_push("replaying...")
	for i in range(5):
		inputs = np.random.random((1, 3, input_size, input_size)).astype(np.float16)
		# inputs = torch.randn((1, 3, input_size, input_size)).cuda().half()
		nvtx.range_push("Singular Replay")
		t1=time.time()
		oginputs[:]=inputs
		# oginputs.copy_(inputs)
		h_input[:] = oginputs.reshape(-1)
		cuda.memcpy_htod_async(d_input, h_input, stream)
		g.replay()
		cuda.memcpy_dtoh_async(h_output, d_output, stream)
		torch.cuda.synchronize()
		nvtx.range_pop()
		time_sum+=time.time()-t1
		if i!=0:
			if (previous_out == h_output).all():
				print("inputs are changing but outputs are not")
		# previous_out=ogoutputs.clone()
		previous_out=np.copy(h_output.copy())
	nvtx.range_pop()
	print("using torchcudagraphsonnxTRT fp16 mode:")
	print("avg cost time: ", round(1000.0*time_sum/5.0,4),'ms')






	# time_sum=0
	# model1=model.float().eval()
	# for i in range(5):
	# 	inputs = torch.tensor(np.random.random((1, 3, input_size, input_size))).cuda().float()
	# 	t1 = time.time()
	# 	# in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(engine)
	# 	out=model1(inputs)
	# 	# print(type(res))
		
	# 	time_sum+=time.time()-t1
	# print("using torch fp32 mode:")
	# print("avg cost time: ", round(1000.0*time_sum/5.0,4),'ms')
	# time_sum=0
	# halfmodel=model.half().eval()
	# for i in range(5):
	# 	inputs = torch.tensor(np.random.random((1, 3, input_size, input_size))).cuda().half()
		
	# 	t1 = time.time()
	# 	# in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(engine)
	# 	out=halfmodel(inputs)
	# 	# print(type(res))
		
	# 	time_sum+=time.time()-t1
	# print("using torch fp16 mode:")
	# print("avg cost time: ", round(1000.0*time_sum/5.0,4),'ms')
	# time_sum=0
	# torch.backends.cudnn.benchmark = True
	# model1=model.float().eval()
	# s = torch.cuda.Stream()
	# torch.cuda.synchronize()
	# inputs = torch.tensor(np.random.random((1, 3, input_size, input_size))).cuda().float()
	# with torch.cuda.stream(s):
	# 	for _ in range(5):
	# 		out=model1(inputs)
	# 	torch.cuda.empty_cache()
	# 	g = torch.cuda._Graph()
	# 	torch.cuda.synchronize()
	# 	g.capture_begin()
	# 	out=model1(inputs)
	# 	g.capture_end()
	# 	torch.cuda.synchronize()

	# for _ in range(5):
	# 	t1=time.time()
	# 	g.replay()
	# 	torch.cuda.synchronize()
		
	# 	time_sum+=time.time()-t1
	# print("using cudagraphsfp32 mode:")
	# print("avg cost time: ", round(1000.0*time_sum/5.0,4),'ms')
	# time_sum=0
	# torch.backends.cudnn.benchmark = True
	# model2=model.half().eval()
	# s = torch.cuda.Stream()
	# torch.cuda.synchronize()
	# inputs = torch.tensor(np.random.random((1, 3, input_size, input_size))).cuda().half()
	# with torch.cuda.stream(s):
	# 	for _ in range(5):
	# 		out=model2(inputs)
	# 	torch.cuda.empty_cache()
	# 	g = torch.cuda._Graph()
	# 	torch.cuda.synchronize()
	# 	g.capture_begin()
	# 	out=model2(inputs)
	# 	g.capture_end()
	# 	torch.cuda.synchronize()

	# for _ in range(5):
	# 	t1=time.time()
	# 	g.replay()
	# 	torch.cuda.synchronize()
		
	# 	time_sum+=time.time()-t1
	# print("using cudagraphsfp16 mode:")
	# print("avg cost time: ", round(1000.0*time_sum/5.0,4),'ms')
	# torch._C._jit_set_nvfuser_enabled(True)
	# torch._C._jit_set_texpr_fuser_enabled(False)
	# torch._C._jit_set_profiling_executor(True)
	# torch._C._jit_set_profiling_mode(True)
	# torch._C._jit_override_can_fuse_on_cpu(False)
	# torch._C._jit_override_can_fuse_on_gpu(False)
	# torch._C._jit_set_bailout_depth(20)
	# time_sum=0
	# torch.backends.cudnn.benchmark = True
	# model1=torch.jit.script(timm.create_model('mixnet_m', pretrained=False, scriptable=True).cuda()).float().eval()
	# s = torch.cuda.Stream()
	# torch.cuda.synchronize()
	# inputs = torch.tensor(np.random.random((1, 3, input_size, input_size))).cuda().float()
	# with torch.cuda.stream(s):
	# 	for _ in range(5):
	# 		out=model1(inputs)
	# 	torch.cuda.empty_cache()
	# 	g = torch.cuda._Graph()
	# 	torch.cuda.synchronize()
	# 	g.capture_begin()
	# 	out=model1(inputs)
	# 	g.capture_end()
	# 	torch.cuda.synchronize()

	# for _ in range(5):
	# 	t1=time.time()
	# 	g.replay()
	# 	torch.cuda.synchronize()
		
	# 	time_sum+=time.time()-t1
	# print("using nvfusedcudagraphsfp32 mode:")
	# print("avg cost time: ", round(1000.0*time_sum/5.0,4),'ms')
	# time_sum=0
	# torch.backends.cudnn.benchmark = True
	# model2=torch.jit.script(timm.create_model('mixnet_m', pretrained=False, scriptable=True).cuda()).half().eval()
	# s = torch.cuda.Stream()
	# torch.cuda.synchronize()
	# inputs = torch.tensor(np.random.random((1, 3, input_size, input_size))).cuda().half()
	# with torch.cuda.stream(s):
	# 	for _ in range(5):
	# 		out=model2(inputs)
	# 	torch.cuda.empty_cache()
	# 	g = torch.cuda._Graph()
	# 	torch.cuda.synchronize()
	# 	g.capture_begin()
	# 	out=model2(inputs)
	# 	g.capture_end()
	# 	torch.cuda.synchronize()

	# for _ in range(5):
	# 	t1=time.time()
	# 	g.replay()
	# 	torch.cuda.synchronize()
		
	# 	time_sum+=time.time()-t1
	# print("using nvfusedcudagraphsfp16 mode:")
	# print("avg cost time: ", round(1000.0*time_sum/5.0,4),'ms')



