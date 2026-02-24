import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("coneslayer-simplified.onnx",
                            providers=["CPUExecutionProvider"])

print("Inputs:")
for i in sess.get_inputs():
    print(i.name, i.shape)

print("\nOutputs:")
for o in sess.get_outputs():
    print(o.name, o.shape)
