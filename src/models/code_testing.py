from multitask import MTLArchitecture

model = MTLArchitecture(
    input_dim= 3800,
    latent_dim= 128,
    num_classes= 7
)

for name, module in model.named_modules():
    pass