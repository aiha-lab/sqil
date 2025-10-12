from torch.utils.data import IterableDataset

class IndexedIterableDataset(IterableDataset):
    def __init__(self, base_ds):
        self.base = base_ds  # RLDSDataset (Iterable)
    def __iter__(self):
        for idx, item in enumerate(self.base):
            item["uid"] = item.get("uid", str(idx))
            yield item
