from torchview import draw_graph 
import torch
model = torch.load('data/model/model_torch/model_torch_vgg19.pth')
batch_size = 32
# device='meta' -> no memory is consumed for visualization 
model_graph = draw_graph(model, input_size=(batch_size, 64), device='meta') 
model_graph.visual_graph