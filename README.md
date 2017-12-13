# PILAE
Pseduo-inverse Learning algorithm for Auto-encoder

### Project content
\PILAE
    \data
    \dataset # path to dataset
        \mnist
        \fashion_mnist
        \cifar10
    \eps
    \log #path to save the param & result of the code
    \src
        __init__.py
        autoencoder_keras.py
        cnn_keras.py
        col_PILAE.py
        Hog.py
        IPILAE.py
        row_PILAE.py
        test.py
        tools.py
    README.md

### Dataset Downloads

baidu pan url: https://pan.baidu.com/s/1c1HyfUK code: 2aig

### How to use?
```python
import src.row_PILAE as rp
pilae = rp.PILAE(k=[0.78, 0.43], pilk=0.07, alpha=[0.8, 0.7], pil_p = [2000, 1000], AE_layer=1, PIL_layer=2, activeFunc='sig')
pilae.fit(X_train, y_train)

```
where k is the list type param denoting the regularization param of the PILAE, pilk is the float type param denoting of regularization param of the PIL, alpha is the list type param denoting the \alpha, pil_p is the list type param of the PIL, AE_layer is the int type param of PILAE denote layer of the PILAE, PIL_layer is the int type param of PIL denote layer of the PIL. All the list type param corresponds to the each layer's param.

The name of all the params correspond to the name of the variable in the paper.


