import numpy as np
import torch

class LabelBuffer:
    def __init__(self, max_size=200000):
        self.max_size = max_size
        self.state = np.zeros((max_size, 8), dtype=np.float32)
        self.pixel = np.zeros((max_size, 64, 64, 3), dtype=np.uint8)
        self.action = np.zeros((max_size, 32), dtype=np.float32)
        self.label = np.zeros((max_size,), dtype=np.int32)
        self.size = 0          
        self.ptr = 0           

    def add(self, state, pixel, action, result):
        self.pixel[self.ptr] = pixel
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.label[self.ptr] = result
        self.ptr = (self.ptr + 1) % self.max_size
        if self.size < self.max_size:
            self.size += 1

    def __len__(self):
        return self.size

    def sample(self, batch_size):
        if self.size == 0:
            raise ValueError("Buffer is empty")

        labels = self.label[:self.size]
        idx_0 = np.where(labels == 0)[0]
        idx_1 = np.where(labels == 1)[0]

        half = batch_size // 2
        n0 = min(len(idx_0), half)
        n1 = min(len(idx_1), half)

        if n0 < half:
            n1 = min(batch_size - n0, len(idx_1))
        elif n1 < half:
            n0 = min(batch_size - n1, len(idx_0))

        sample_idx_0 = np.random.choice(idx_0, n0, replace=False) if n0 > 0 else np.empty(0, dtype=int)
        sample_idx_1 = np.random.choice(idx_1, n1, replace=False) if n1 > 0 else np.empty(0, dtype=int)

        batch_idx = np.concatenate([sample_idx_0, sample_idx_1])
        np.random.shuffle(batch_idx)
        
        states = torch.from_numpy(self.state[batch_idx]).float()
        pixels = torch.from_numpy(self.pixel[batch_idx]).float()
        actions = torch.from_numpy(self.action[batch_idx]).float()
        labels = torch.from_numpy(self.label[batch_idx]).float()
        return states, pixels, actions, labels

    def iter_batches(self, batch_size, shuffle=True, drop_last=False):
        if self.size == 0:
            raise ValueError("Buffer is empty")
        
        indices = np.arange(self.size)
        if shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, self.size, batch_size):
            end = start + batch_size
            if end > self.size and drop_last:
                break
            batch_idx = indices[start:end]
            states = torch.from_numpy(self.state[batch_idx]).float()
            pixels = torch.from_numpy(self.pixel[batch_idx]).float()
            actions = torch.from_numpy(self.action[batch_idx]).float()
            labels = torch.from_numpy(self.label[batch_idx]).float()
            yield states, pixels, actions, labels