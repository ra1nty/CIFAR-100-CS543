import numpy as np

def CutOut(n, size):
    def _cutout(img):
        h, w = img.size[:2]
        mask = np.ones((h, w, 3), np.float32)
        for n in range(n):
            offh = np.random.randint(h)
            offw = np.random.randint(w)
            x1 = np.clip(offw - size//2, 0, w)
            x2 = np.clip(offw + size//2, 0, w)
            y1 = np.clip(offh - size//2, 0, h)
            y2 = np.clip(offh + size//2, 0, h)
            mask[y1:y2, x1:x2] = [0., 0., 0.]
        img = img * mask
        return img
    
    return _cutout
