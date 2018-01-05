# PILAE
Pseudo-inverse Learning algorithm for training Auto-encoders

### Dataset Downloads

baidu pan url: https://pan.baidu.com/s/1c1HyfUK code: 2aig

### How to use?
```python
from src.pilae import PILAE

(X_train, y_train), (_, _) = load_dataset()

ae_k_list = [0.78, 0.85, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
pil_k = 0.03
alpha_list = [0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
pil_p = [400, 300]
pilae = PILAE(ae_k_list=ae_k_list, pil_p=pil_p, pil_k=pil_k, alpha=alpha_list, ae_layers=10, pil_layers=0, acFunc='sig')
pilae.train_pilae(X_train, y_train)

```
where ae_k_list is the list type param denoting the regularization param of the PILAE, pil_k is the float type param denoting of regularization param of the PIL, alpha is the list type param denoting the \alpha, pil_p is the list type param of the PIL, ae_layers is the int type param of PILAE denote layer of the PILAE, PIL_layer is the int type param of PIL denote layer of the PIL. All the list type param corresponds to the each layer's param.

The name of all the params correspond to the name of the variable in the paper SMC 2017 K. Wang et al.


