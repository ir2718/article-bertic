import torch
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

def get_optimizer(optimizer_type, model, lr, **optimizer_kwargs):
    d = {
        "adamw": torch.optim.AdamW,
        "adam": torch.optim.Adam
    }
    
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": optimizer_kwargs["weight_decay"] if "weight_decay" in optimizer_kwargs else 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    optimizer = d[optimizer_type](optimizer_grouped_parameters, lr=lr)
    
    return optimizer

def get_scheduler(scheduler_type, optimizer, num_training_steps, warmup_ratio):
    d = {
        "linear": get_linear_schedule_with_warmup,
        "cosine": get_cosine_schedule_with_warmup,
    }
    
    scheduler = d[scheduler_type](
        optimizer=optimizer, 
        num_training_steps=num_training_steps,
        num_warmup_steps=int(warmup_ratio * num_training_steps)    
    )
    
    return scheduler