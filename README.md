# CUR Decomposition
Dimensionality reduction using CUR decomposition ($A\approx CUR$)

### Parameters

- **A** : *numpy.array, default=None*, n x m dataset matrix (n: number of features, m: number of instances)
- **n_components** : *int, default=2*, Desired dimensionality of output data
- **sampling** : *str, default='random'*, Sampling method for column and row selection
- **n_iter** : *int, default=5*, Number of iterations for CUR approximations


### Attributes

- **C** : *numpy.array*, Matrix containing selected columns of A
- **Cpinv** : *numpy.array*, Pseudo inverse of C
- **R** : *numpy.array*, Matrix containing selected rows of A    
- **Rpinv** : *numpy.array*, Pseudo inverse of R
- **U** : *numpy.array*, U = Cpinv @ A @ Rpinv
- **colidx** : *list*, List of selected column indices for constructing C
- **rowidx** : *list*, List of selected row indices for constructing R

### Example
```python
import numpy as np
from sklearn.datasets import load_wine

n_components = 10
A = load_wine().data.T

cur = CURdecomposition(A=A, n_components=n_components)
cur_error = np.linalg.norm(A - cur.C @ cur.Cpinv @ A)
print(f"[CUR] reconstruction error: {cur_error:.4f}")
```

```
# Output
[CUR] reconstruction error: 6.5836
```
