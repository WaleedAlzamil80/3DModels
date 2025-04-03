def setTrainable(model, n):
    counter = 0
    for param in model.parameters():
        counter += 1
        if(counter > n):
            param.requires_grad = True

def FreezeFirstN(model, n):
    counter = 0
    for param in model.parameters():
        counter += 1
        if(counter < n):
            param.requires_grad = False

def printParamSize(model):
    for param in model.parameters():
        print(param.size())

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
