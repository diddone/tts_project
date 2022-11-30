from .collator import Collator
from .buffer import get_data_to_buffer, BufferDataset
from torch.utils.data import Dataset, DataLoader

def get_training_loader(train_config):

    buffer = get_data_to_buffer(train_config)

    dataset = BufferDataset(buffer)
    collator = Collator(train_config.batch_expand_size)

    training_loader = DataLoader(
        dataset,
        batch_size=train_config.batch_expand_size * train_config.batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
        num_workers=0
    )

    return training_loader


