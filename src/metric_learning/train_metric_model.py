from src.metric_learning.parser import parse
from src.metric_learning.similarity_dataset import get_dataset
from src.metric_learning.metric_model import get_model
from src.metric_learning.train_utils import get_optimizer, get_scheduler
from torch.utils.data import DataLoader
from transformers import set_seed

args = parse()
set_seed(args.seed)

train_dataset = get_dataset(loss=args.loss_function, root=args.dataset_root, split="train")
train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

val_dataset = get_dataset(loss=args.loss_function, root=args.dataset_root, split="validation")
val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size)

test_dataset = get_dataset(loss=args.loss_function, root=args.dataset_root, split="test")
test_dataloader = DataLoader(test_dataset, batch_size=args.val_batch_size)

model = get_model(
    loss=args.loss_function,
    model_name=args.model_name,
    embedding_size=args.embedding_size,
    device=args.device,
    pooling_type=args.pooling_type,
)

optimizer = get_optimizer(
    optimizer_type=args.optimizer,
    model=model,
    lr=args.lr,
    weight_decay=args.weight_decay
)

scheduler = get_scheduler(
    scheduler_type=args.scheduler,
    optimizer=optimizer,
    num_training_steps=(len(train_dataloader) // args.gradient_accumulation_steps),
    warmup_ratio=args.warmup_ratio
)

print()
print("Starting training . . .")

model.train_model(
    num_epochs=args.num_epochs,
    train_dataloader=train_dataloader, 
    validation_dataloader=val_dataloader,
    optimizer=optimizer, 
    scheduler=scheduler,
    gradient_accumulation_steps=args.gradient_accumulation_steps
)

model.test_model(
    test_dataloader=test_dataloader
)
model.save_all_metrics()