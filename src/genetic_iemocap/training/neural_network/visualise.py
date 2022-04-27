import torch
from torchviz import make_dot
from genetic_iemocap.models.iemocap_baseline import Baseline

model = Baseline(5)
x = torch.randn((32,5,10))
y = model(x)

make_dot(y, params=dict(list(model.named_parameters()))).render("baseline_model", format="png")