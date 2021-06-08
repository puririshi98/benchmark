import torch
import torch.cuda as cuda
import torchvision.models as models
import time
import tensorrt as trt

dummy_input = torch.randn(1, 3, 224, 224).cuda()
model = models.resnet18().cuda()
torch.onnx.export(model, dummy_input, "res18.onnx", verbose=True)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ONNX_MODEL = "res18.onnx"

with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config() as config, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
	# Configure the builder here.
	config.max_workspace_size = 2**30
	builder.max_workspace_size = 1 << 20
    builder.max_batch_size = 1
	# Parse the model to create a network.
	with open(ONNX_MODEL, 'rb') as model:
		parser.parse(model.read())
	# Build and return the engine. Note that the builder, network and parser are destroyed when this function returns.
	engine=builder.build_engine(network, config)
	context=engine.create_execution_context()
	since=time.time()
	for i in range(10):
		# Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
		h_input = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
		h_output = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)
		# Allocate device memory for inputs and outputs.
		d_input = cuda.mem_alloc(h_input.nbytes)
		d_output = cuda.mem_alloc(h_output.nbytes)
		# Create a stream in which to copy inputs/outputs and run inference.
		stream = cuda.Stream()
		# Transfer input data to the GPU.
		cuda.memcpy_htod_async(d_input, h_input, stream)
		# Run inference.
		context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
		# Transfer predictions back from the GPU.
		cuda.memcpy_dtoh_async(h_output, d_output, stream)
		# Synchronize the stream
		stream.synchronize()
print("Avg iter time:",round(time.time()-since,2),"seconds")
				