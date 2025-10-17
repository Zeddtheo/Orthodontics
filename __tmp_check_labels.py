import sys
sys.path.append("src/iMeshSegNet")
import torch
import m0_dataset
cfg = m0_dataset.DataConfig()
cfg.label_mode = "single_arch_16"
cfg.batch_size = 1
cfg.pin_memory = False
train_loader, _ = m0_dataset.get_dataloaders(cfg)
uniques = set()
for idx, batch in enumerate(train_loader):
    labels = batch[1]
    uniq = torch.unique(labels).tolist()
    uniques.update(uniq)
    print(f"Batch {idx} unique: {sorted(uniq)}")
    if idx >= 4:
        break
print("Union across first 5 batches:", sorted(uniques))
