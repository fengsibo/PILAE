# PILAE
Pseudo-inverse Learning algorithm for training Auto-encoders

### Dataset Downloads

链接: https://pan.baidu.com/s/1b5XHm2NjrLDQlvZaLWP9rw 密码: zrfr

下载后放在工程根目录

### How to use?

具体可以参考测试文件test.py 里面有使用方法

```python
from src.pilae import PILAE
import src.tools as tools

(X_train, y_train), (X_test, y_test) = tools.load_npz(DATASET_PATH)
X_train = X_train.reshape(-1, DIMESION_OF_DATASET).astype('float64') / 255.
X_test = X_test.reshape(-1, DIMESION_OF_DATASET).astype('float64') / 255.

# 创建对象时注意 num_*_layers和len(list)和len(pil*_p)对应(层数和list长度对应)
pilae = PILAE(pilae_p=[500, 480, 460],
              pil_p=[300],
              ae_k_list=[0.7, 0.1, 0.1],
              pil_k=0.0,
              acFunc='sig')

pilae.train_pilae(X_train, y_train)

```
### param:
ae_k_list:list类型参数 list的引索的值是对应某层的k(k为正则化系数)的值

pilae_p:list类型参数 list的引索的值对应pilae某层的p(p为隐层单元个数)的值

pil_p:list类型参数 list的引索的值对应mlp分类器某层的p(p为隐层单元个数)的值

pil_k:float类型参数 代表mlp的正则化系数

alpha:经验公式给出的参数(这个版本代码没有用到，保留)

acFunc:激活函数  'sig', 'sin', 'srelu', 'tanh', 'swish', 'relu' 可选


### 说明

传参时确定好pilae_p和pil_p两个list, 手动设置隐层节点数 

同时ae正则化参数k_list要和pilae_p长度相同 

层数由传入的list的长度决定

具体参数描述请参考论文 [SMC 2017 K. Wang et al](https://www.researchgate.net/profile/Ping_Guo3/publication/320077277_Autoencoder_Low_Rank_Approximation_and_Pseudoinverse_Learning_Algorithm/links/59ccc36d45851556e98792db/Autoencoder-Low-Rank-Approximation-and-Pseudoinverse-Learning-Algorithm.pdf).

[注] 如果有幸你用了我的代码，请将该github链接添加到论文的引用
