import numpy as np

class CutOut(object):
    """Cutout an image tensor image with n holes with size length.
    """
    def __init__(self, n, length):
        self.n = n
        self.length = length
    
    def __call__(self, tensor):
        _, h, w = tensor.shape
        mask = np.ones((3, h, w), dtype=np.float32)
        for i in range(self.n):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[:, y1: y2, x1: x2] = 0.
        tensor = tensor * torch.from_numpy(mask)
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(n={0}, length={1})'.format(self.n, self.length)
