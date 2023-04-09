import numpy as np

class DataIterator:
    def __init__(self, dataset, batch_size, shuffle) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seq = list(range(len(self.dataset)))
        self.pivot = 0

        if shuffle:
            np.random.shuffle(self.seq)

    def __iter__(self):
        return self

    def __next__(self):
        if self.pivot >= len(self.seq):
            raise StopIteration()
        next_pivot = self.pivot + self.batch_size
        current = [self.dataset[i] for i in self.seq[self.pivot : next_pivot]]
        self.pivot = next_pivot
        return tuple(np.stack(x) for x in zip(*current))

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        return DataIterator(self.dataset, self.batch_size, self.shuffle)

class MNISTDataset:
    def __init__(self, data_path, mode):
        if mode not in ['train', 'test']:
            raise ValueError('invalid mode!')
        self.images = np.load(data_path + mode + '_images.npy')
        self.labels = np.load(data_path + mode + '_labels.npy')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

class PartialDataset:
    def __init__(self, source, slice) -> None:
        self.source = source
        self.slice = slice

    def __len__(self):
        return len(self.slice)

    def __getitem__(self, index):
        return self.source[self.slice[index]]



def random_split(dataset, ratio):
    total_size = len(dataset)
    random_index = list(range(total_size))
    np.random.shuffle(random_index)
    all_index_list = []
    pivot = 0

    for r in ratio:
        next_pivot = min(total_size, pivot + round(r * total_size))
        all_index_list.append(random_index[pivot:next_pivot])
        pivot = next_pivot

    return tuple(PartialDataset(dataset, slice) for slice in all_index_list)