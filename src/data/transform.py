from __future__ import annotations

import albumentations as A
import numpy as np

image = np.ones((224, 224, 3))
mask = np.ones((224, 224, 2))
mask[:, :, 0] = 0

transformed = A.RandomCrop(width=100, height=100)(image=image, mask=mask)

print(transformed['image'].shape)
