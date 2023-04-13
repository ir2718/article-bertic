import torch
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

def get_optimizer(optimizer_type, model, **optimizer_kwargs):
    d = {
        "adamw": torch.optim.AdamW,
        "adam": torch.optim.Adam
    }
    
    
    if "weight_decay" in optimizer_kwargs:
        param_optimizer = list(model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': optimizer_kwargs["weight_decay"]},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    else:
        optimizer_parameters =list(model.named_parameters())

    optimizer = d[optimizer_type](optimizer_parameters, **optimizer_kwargs)
    
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