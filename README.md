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

###How to use?
```python
import src.row_PILAE as rp
pilae = rp.PILAE(k=k_list, pilk=pilk, alpha=alpha_list, pil_p = pil_p, AE_layer=1, PIL_layer=2, activeFunc='sig')
pilae.fit(X_train, y_train)

```
where k is the list type param of PILAE, pilk is the list type param of PIL, pil_p is the list type param of the PIL, AE_layer is the int type param of PILAE denote layer of the PILAE, PIL_layer is the int type param of PIL denote layer of the PIL.

The name of all the params correspond to the name of the vadiable in the paper.


