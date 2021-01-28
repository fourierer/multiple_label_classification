# multiple_label_classification
multiple label classification with MobileNetV2, ShuffleNetV2 and GhostNet



1.多元分类和多标签分类的比较

多元分类是将一个样本分到一个特定的类别当中，而多标签分类是将一个样本分到多个类别当中（一个样本可能属于多个类别），所以损失函数会不一样，其次数据加载也不一样。

（1）损失函数

```python
import torch.nn as nn
loss1 = nn.CrossEntropyLoss() # 多元分类损失函数
loss2 = nn.BCEWithLogitsLoss() # 多标签分类损失函数
```

**多元分类：**

网络最后一层的输出先经过softmax函数，使得每一个元素值映射到一个概率，再对概率取对数，最后将前面的输出与Label对应的那个值取负数就得到了该样本的loss（如果有batch的话要需要取均值，且最后一步实际上就是NLLLOSS函数），将这三步合成一步就形成CrossEntropyloss函数，公式如下：
$$
loss(x,class) = -ln(\frac{exp(x[class])}{\sum_jexp(x[j])})=-x[class]+ln(\sum_jexp(x[j]))
$$
从公式当中可以直接看出来三个函数nn.Softmax()，torch.log()，nn.NLLLoss()的作用顺序，并且loss函数并没有像交叉熵函数那样计算$-[t_iln(y_i)+(1-t_i)ln(1-y_i)]$两项，实际上只取了前一项。

代码验证：

```python
import torch
import torch.nn as nn

# 分布调用函数计算损失
input = torch.randn(3,3)
input
'''
tensor([[-0.9147, -1.8131, -0.6992],
        [-1.1762,  1.1391,  1.7756],
        [-0.1639,  0.2338,  0.0073]])
'''
sm = nn.Softmax(dim=1)
sm(input)
'''
dim=1将每一行作为一个样本来计算，从网络输出映射到概率，默认dim=1
tensor([[0.3777, 0.1538, 0.4685],
        [0.0330, 0.3346, 0.6324],
        [0.2721, 0.4050, 0.3229]])
'''
torch.log(sm(input))
'''
tensor([[-0.9737, -1.8721, -0.7582],
        [-3.4101, -1.0948, -0.4583],
        [-1.3016, -0.9039, -1.1304]])
'''
loss = nn.NLLLoss()
target = torch.tensor([0,2,1])
loss(torch.log(sm(input)), target)
'''
根据标签选择计算的值，第一个样本选第0个概率取负数，为0.9737；第二个样本选第2个概率取负数，为0.4583；第三个样本选择第1个概率值取负数，为0.9039；三者取均值，得到0.7786；
tensor(0.7786)
'''

# 直接调用nn.CrossEntropyLoss()计算损失
loss = nn.CrossEntropyLoss()
loss(torch.log(sm(input)), target)
'''
tensor(0.7786)，与分步计算相同
'''
```



**多标签分类：**

网络最后一层的输出先经过sigmoid函数（网络输出的向量经过sigmoid函数，第$i$个值表示该样本属于第$i$类的概率，此时向量的值加和不为1），再经过交叉熵函数来计算loss，公式如下：
$$
loss(o,t) = -\frac{1}{n}\sum_i(t[i]*ln(o[i])+(1-t[i])*log(1-o[i]))
$$
此时的损失计算函数为nn.BCELoss()里面包括了两项，而且是对sigmoid和CrossEntropyloss函数不一样。

代码验证：

```python
import torch
import torch.nn as nn
import math

# 分步计算损失
input = torch.randn(3,3)
input
'''
tensor([[ 0.0754,  0.3111,  0.7609],
        [-0.6940, -1.2521,  0.5681],
        [-2.0807,  0.0732, -2.3314]])
'''
si = nn.Sigmoid()
si(input)
'''
tensor([[0.5189, 0.5772, 0.6816],
        [0.3331, 0.2223, 0.6383],
        [0.1110, 0.5183, 0.0886]])
'''
# 假设标签如下
target = torch.tensor([[0,1,1],[1,1,1],[0,0,0]])

r11 = 0 * math.log(0.8707) + (1-0) * math.log((1 - 0.8707))
r12 = 1 * math.log(0.7517) + (1-1) * math.log((1 - 0.7517))
r13 = 1 * math.log(0.8162) + (1-1) * math.log((1 - 0.8162))

r21 = 1 * math.log(0.3411) + (1-1) * math.log((1 - 0.3411))
r22 = 1 * math.log(0.4872) + (1-1) * math.log((1 - 0.4872))
r23 = 1 * math.log(0.6815) + (1-1) * math.log((1 - 0.6815))

r31 = 0 * math.log(0.4847) + (1-0) * math.log((1 - 0.4847))
r32 = 0 * math.log(0.6589) + (1-0) * math.log((1 - 0.6589))
r33 = 0 * math.log(0.5273) + (1-0) * math.log((1 - 0.5273))

r1 = -(r11 + r12 + r13) / 3
#0.8447112733378236
r2 = -(r21 + r22 + r23) / 3
#0.7260397266631787
r3 = -(r31 + r32 + r33) / 3
#0.8292933181294807
bceloss = (r1 + r2 + r3) / 3 
print(bceloss)
'''
0.8000147727101611
'''

# 使用BCEloss函数
loss = nn.BCELoss()
print(loss(m(input), target))
'''
tensor(0.8000)
'''

# 使用BCEWithLogitsLoss函数
loss = nn.BCEWithLogitsLoss()
print(loss(input, target))
'''
tensor(0.8000)
'''
```





（2）准确率指标衡量

多元分类：

准确率有top-k的划分，前k类有预测对的都将该样本视为正确分类。代码如下：

```python
def accuracy1(output, target, topk=(1, 5)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
```



多标签分类：（多标签分类的指标较复杂，可以取简单的一种）

对于每个样本，要预测多个标签，输出值经过sigmoid函数之后设置阈值，使得大于阈值的设为1，小于阈值的设为0，这样就得到一个样本的预测值。对比预测值和标签向量，将同时为1的类别个数记为$N$，将预测值向量中1的个数记为$M_1$，将标签向量中1的个数记为$M_2$。$\frac{N}{M_1}$称为准确率，即模型预测出来的有多少是正确的；$\frac{N}{M_2}$称为召回率，即正确的类别可以召回来多少，用这两个指标来衡量每个epoch的模型效果。

代码如下：

```python
def accuracy(output, target):
    acc = recall = 0
    pred = torch.sigmoid(output).ge(0.5).cpu()
    pred = [pred_one.nonzero().reshape(-1).numpy().tolist() for pred_one in pred]
    target = target.cpu()
    target = [target_one.nonzero().reshape(-1).numpy().tolist() for target_one in target]
    for i in range(len(target)):
        #print(pred[i])
        #print(target[i])
        if len(pred[i]) != 0:
            acc += len(set(pred[i])&set(target[i])) / (len(pred[i])+1e-7)
            recall += len(set(pred[i])&set(target[i])) / (len(target[i])+1e-7)
        acc = acc / len(target)
        recall = recall / len(target)
        return torch.tensor(acc).cuda().float(), torch.tensor(recall).cuda().float()

```

由于输入的output以及target均在gpu上，在使用这两个量来计算准确率和召回率的时候需要先放到cpu上才可以进行numpy或者list操作，在计算结束后再放回gpu上。



测试两个准确度函数：

```pyhthon
import torch
import numpy as np

def accuracy1(output, target, topk=(1, 2)):
    maxk = max(topk)
    print('maxk:' + str(maxk))
    batch_size = target.size(0)
    print('batch_size:' + str(batch_size))
    _, pred = output.topk(maxk, 1, True, True) # 按照维度1的方向，返回前maxk个最大值对应的类别
    print('pred:' + str(pred))
    pred = pred.t()
    print('pred:' + str(pred))
    print('target_view:' + str(target.view(1, -1)))
    print('target_view_expand:' + str(target.view(1, -1).expand_as(pred)))
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    print('correct:' + str(correct))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

'''
def accuracy2(outputs, targets, threshold):
    acc = 0
    recall = 0
    pred = torch.sigmoid(outputs)
    print('pred:' + str(pred))
    pred = pred.ge(threshold).float()
    print('pred:' + str(pred))
    print('targets.view_as(pred):' + str(targets.view_as(pred)))
    print('pred.eq(targets.view_as(pred)):' + str(pred.eq(targets.view_as(pred))))
    return pred.eq(targets.view_as(pred)).float().mean()
'''
def accuracy2(output, target):
    acc = recall = 0
    pred = torch.sigmoid(output).ge(0.5)
    print('pred:' + str(pred))
    print('pred[1].nonzero():' + str(pred[1].nonzero()))
    print('pred[1].nonzero().reshape():' + str(pred[1].nonzero().reshape(-1)))
    pred = [pred_one.nonzero().reshape(-1).numpy().tolist() for pred_one in pred]
    target = [target_one.nonzero().reshape(-1).numpy().tolist() for target_one in target]
    print('pred:' + str(pred))
    print('target:' + str(target))
    for i in range(len(target)):
        #print(pred[i])
        #print(target[i])
        if len(pred[i]) != 0:
            print('set:' + str(set(pred[i])&set(target[i])))
            acc += len(set(pred[i])&set(target[i])) / (len(pred[i]) + 1e-7)
            recall += len(set(pred[i])&set(target[i])) / (len(target[i])+1e-7)
    acc = acc / len(target)
    recall = recall / len(target)
    return torch.tensor(acc).float(), torch.tensor(recall).float()

inputs = torch.tensor([[ 0.0754,  0.3111,  0.7609],
                      [-0.6940, -1.2521,  0.5681],
                      [-2.0807,  0.0732, -2.3314]])

label1 = torch.tensor([2, 2, 0]) # 多元分类并不是one-hot形式，每个样本的标签就是一个数值

label2 = torch.tensor([[0., 0., 1.],
                       [0., 0., 0.],
                       [0., 0., 0.]])

acc1 = accuracy1(inputs, label1, (1, 2))
print('acc:' + str(acc1))

#label2 = label2.float()
acc2, recall = accuracy2(inputs, label2)
print('acc2:' + str(acc2))
print('recall:' + str(recall))

```

多元分类和多标签分类的ImageFolder返回的标签不一样，多元分类的标签每个类别只需要返回一个数，即类别序号；多标签分类是返回一个向量，属于哪个类别即在哪个位置置1。



（3）数据加载

多元分类：

多元分类任务里面，每一张图片属于一个特定的类别，放在一个类别文件当中，然后使用pytorch中的datasets.ImageFolder去读数据集，对数据进行预处理同时返回标签；在多标签分类中，需要重新写ImageFolder函数。

先给出pytorch官方提供的datasets.ImageFolder源代码，pytorch版本为1.0：( https://pytorch.org/docs/1.2.0/_modules/torch/utils/data/dataloader.html#DataLoader )

```python
import torch.utils.data as data

from PIL import Image

import os
import os.path


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))] # 数据集中的所有文件类别，列表形式
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))} # 类别名字到序号，字典形式
    return classes, class_to_idx

def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir) # 去除路径中的～/
    for target in sorted(os.listdir(dir)): # target为类别名称，如dog
        d = os.path.join(dir, target) # 类别路径，如xx/train/dog/
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames): # 图片名称排序
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname) # 图片路径，如xx/train/dog/1.jpg
                    item = (path, class_to_idx[target]) # 路径和所属类别的序号
                    images.append(item)

    return images # image是一个列表，元素都是二元组，存储图片和图片所属类别的序号

class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """
    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        classes, class_to_idx = find_classes(root) # classes为列表，存储类别名字；class_to_idx为字典，存储列表名字和对应的序号
        samples = make_dataset(root, class_to_idx, extensions) # sample是一个列表，元素都是二元组，存储图片和图片所属类别的序号
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index] # 图片路径和所属的类别的序号
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples
```



多标签分类：

多标签分类一般数据集需要自己整理，只有list文件用于存储类别名称和数据图片的信息，包括url和所属的类别，处理步骤如下：

1）拆分数据集为训练集和测试集

由于所有的数据都存在一个list文件里面，先读取list，再将所有图片分为训练集和测试集两部分，前350000张为测试集，剩下的接近100000张为测试集，给每张图片加上前缀路径分别存储为train.txt和test.txt。list文件格式和划分代码如下：

```python
scenery/00001002.jpg 7 scenery outdoor clouds sun tree sunset sky
selfie/selfie_part1/018DD09A-7D10-4FA3-230F-31E882A2C77020170215_00.jpg 3 person selfie outdoor
nus-wide/dataset/un_building/0348_2274244083.jpg 1 outdoor
nus-wide/dataset/throwing/0060_1477493305.jpg 0
anotated_0106/data1/25EAEDD8-38D5-522B-0D90-52A7592E77DC20171226_02.jpg 4 plants person outdoor tree
others/cizhuan/718ffa03-66ad-11e7-bdeb-2c4d544f3e490.jpg 0
nus-wide/dataset/football/0567_2391156357.jpg 7 scenery buildings outdoor clouds sky cityscape road
nus-wide/dataset/white-tailed/0217_1501598065.jpg 5 clouds outdoor sky birds animal
nus-wide/dataset/farms/0502_166859912.jpg 3 horses outdoor animal
......
```



```python
# 此脚本用于将所有图片划分测试集和训练集

#import json

source_path = 'all-train-image-labels-en_labels_shuf.list'
train_path = 'train.txt'
test_path = 'test.txt'
presuffix = '/mnt/data1/sun.zheng/all-train-val/'

count = 0 # 训练集数据个数，总数据量为448446
train_count = 350000 # 设置训练集数量为350000

with open(train_path, 'w') as f:
    for line in open(source_path):
        line = line.strip('\n') #去掉换行符\n
        line = presuffix + line
        f.write(line)
        f.write('\n')
        #print(line)
        #print(type(line)) # str
        count += 1
        print('current train count:' + str(count))
        if count > train_count:
            break

count = 0 # 当前数量重新置0
with open(test_path, 'w') as f:
    for line in open(source_path):
        count += 1
        print('current test count:' + str(count))
        if count > train_count:
            line = line.strip('\n') # 去掉换行符\n
            line = presuffix + line
            f.write(line)
            f.write('\n')
```



2）将类别名称数字化

标签类别文件为英文名称以及中文释义，可以按照从上往下的顺序依次索引为0，1，2，...，并存为label.txt文件，标签list文件格式以及代码如下：

```shell
clouds,云
coral,珊瑚
cow,牛
dog,狗
fire,火
fish,鱼
flowers,花卉
food,食物
fox,狐狸
garden,花园
glacier,冰川
grass,草
harbor,港口
......
```



```python
import numpy as np
count = 0
with open('valid-labels-107.txt','r',encoding='utf-8') as f:
    dic=[]
    for line in f.readlines():
        line=line.strip('\n') #去掉换行符\n
        b=line.split(',') #将每一行以逗号为分隔符转换成列表
        #print(type(b))
        #print(b)
        del b[1] # 删除第二个元素
        b.append(count) # 增加value
        print(b)
        dic.append(b)
        count += 1
    dic=dict(dic)
print(dic)

# 将字典写入txt文件中
with open('label.txt', 'w') as f:
    for key, value in dic.items():
        f.write(key)
        f.write(' ')
        f.write(str(value)) # 此处必须要加str，否则无法写入
        f.write('\n')
```

这里生成的label.txt后续没有使用，但是代码后续会用。



3）重写datasets.ImageFolder源码，使得适应多标签的数据

在数据文件准备好之后，可以开始重写datasets.ImageFolder类，先给出重写之后的代码：

```python
import torch.utils.data as data
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import os.path

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))] # 数据集中的所有文件类别，列表形式
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))} # 类别名字到序号，字典形式
    return classes, class_to_idx

'''
def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir) # 去除路径中的～/
    for target in sorted(os.listdir(dir)): # target为类别名称，如dog
        d = os.path.join(dir, target) # 类别路径，如xx/train/dog/
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames): # 图片名称排序
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname) # 图片路径，如xx/train/dog/1.jpg
                    item = (path, class_to_idx[target]) # 路径和所属类别的序号
                    images.append(item)

    return images # image是一个列表，元素都是二元组，存储图片和图片所属类别的序号
'''

def make_label(file_path):
    count = 0
    with open(file_path,'r',encoding='utf-8') as f:
        dic=[]
        for line in f.readlines():
            line=line.strip('\n') #去掉换行符\n
            b=line.split(',') #将每一行以逗号为分隔符转换成列表
            #print(type(b))
            #print(b)
            del b[1] # 删除第二个元素
            b.append(count) # 增加value
            #print(b)
            dic.append(b)
            count += 1
        dic=dict(dic)
    return dic

def make_dataset(dir, dic):
    images = []
    # 读取list文件，返回images，和多元分类一样，image是一个列表，元素都是二元组，存储图片和图片所属类别的np.array数组
    for line in open(dir):
        line=line.strip('\n') #去掉换行符\n
        b=line.split(' ') #将每一行以空格为分隔符转换成列表
        length = len(b) # 获取列表b的长度
        # 获取图片路径
        path = b[0]

        # 获取图片所属类别的np.array数组
        label_array = np.zeros(107) # 全0的标签数组

        for i in range(length - 2):
            label_array[dic[b[i + 2]]] = 1 #所属的类别置1
        item = (path, label_array)
        images.append(item)

    return images

class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """
    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        #classes, class_to_idx = find_classes(root) # classes为列表，存储类别名字；class_to_idx为字典，存储列表名字和对应的序号
        #samples = make_dataset(root, class_to_idx, extensions) # sample是一个列表，元素都是二元组，存储图片和图片所属类别的序号
        file_path = '/home/momo/sun.zheng/multi_label_classification/GhostNet/valid-labels-107.txt'
        dic = make_label(file_path)
        samples = make_dataset(root, dic) # 传入一个list文件路径和标签字典，从该list文件中解析出所有图片和相应的标签
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        #self.classes = classes
        #self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index] # 图片路径和所属的类别的np.array数组
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target # 返回loader(path)和该图片标签的一维np.array数组

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str



IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples
```

​    ImageFolder类继承了DatasetFolder类，参数有root数据根目录，图片加载方式loader，图片预处理方式为transform，相应的标签预处理为target_transform(target_transform默认为None)。在实例化ImageFolder类对象时，要先初始化DatasetFolder类。

​    DatasetFolder初始化函数__init__中，传入原始标签的路径，make_label函数将原始标签转换为一个字典，键为类别的英文名称，值为类别的数组索引；再将得到标签字典和训练集（测试集）对应的txt文件路径传入make_dataset函数，得到数据集的二元组，\__init__函数，make_label以及make_dataset函数源码如下：

```python
    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        #classes, class_to_idx = find_classes(root) # classes为列表，存储类别名字；class_to_idx为字典，存储列表名字和对应的序号
        #samples = make_dataset(root, class_to_idx, extensions) # sample是一个列表，元素都是二元组，存储图片和图片所属类别的序号
        file_path = '/home/momo/sun.zheng/multi_label_classification/GhostNet/valid-labels-107.txt'
        dic = make_label(file_path)
        samples = make_dataset(root, dic) # 传入一个list文件路径和标签字典，从该list文件中解析出所有图片和相应的标签
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        #self.classes = classes
        #self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform
```

```python
def make_label(file_path):
    count = 0
    with open(file_path,'r',encoding='utf-8') as f:
        dic=[]
        for line in f.readlines():
            line=line.strip('\n') #去掉换行符\n
            b=line.split(',') #将每一行以逗号为分隔符转换成列表
            #print(type(b))
            #print(b)
            del b[1] # 删除第二个元素
            b.append(count) # 增加value
            #print(b)
            dic.append(b)
            count += 1
        dic=dict(dic)
    return dic

def make_dataset(dir, dic):
    images = []
    # 读取list文件，返回images，和多元分类一样，images是一个列表，元素都是二元组，存储图片和图片所属类别的np.array数组
    for line in open(dir):
        line=line.strip('\n') #去掉换行符\n
        b=line.split(' ') #将每一行以空格为分隔符转换成列表
        length = len(b) # 获取列表b的长度
        # 获取图片路径
        path = b[0]

        # 获取图片所属类别的np.array数组
        label_array = np.zeros(107) # 全0的标签数组

        for i in range(length - 2):
            label_array[dic[b[i + 2]]] = 1 #所属的类别置1
        item = (path, label_array)
        images.append(item)

    return images

```





2.多标签分类任务训练代码解读

（和多元分类大体一样，不过需要修改一些地方）

（1）导入改写之后的ImageFolder

```python
from D_v2 import ImageFolder # 代替pytorch自带的ImageFolder来读取多标签数据
```

（2）输入的参数更改为训练集和测试集的txt文件

```python
parser.add_argument('--train_data', metavar='DIR', default='/home/momo/sun.zheng/multi_label_classification/GhostNet/train.txt', help='path to train_dataset')
parser.add_argument('--val_data', metavar='DIR', default='/home/momo/sun.zheng/multi_label_classification/GhostNet/test.txt', help='path to val_dataset')
```

（3）使用改写之后的ImageFolder来加载数据

```python
train_dataset = ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=2,
                                               pin_memory=True,
                                               sampler=train_sampler)
val_dataset = ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size = args.batch_size,
                                            shuffle=False,
                                            num_workers=2,
                                            pin_memory=True)

```

（4）衡量指标改为精确度和召回率

```python
top1 = AverageMeter('Acc', ':6.2f')
top2 = AverageMeter('Recall', ':6.2f')
# 后续的相应修改
```

（5）修改准确率函数

```python
def accuracy(output, target):
    acc = recall = 0
    pred = torch.sigmoid(output).ge(0.5).cpu()
    pred = [pred_one.nonzero().reshape(-1).numpy().tolist() for pred_one in pred]
    target = target.cpu()
    target = [target_one.nonzero().reshape(-1).numpy().tolist() for target_one in target]
    for i in range(len(target)):
        if len(pred[i]) != 0:
            acc += len(set(pred[i])&set(target[i])) / (len(pred[i])+1e-7)
            recall += len(set(pred[i])&set(target[i])) / (len(target[i])+1e-7)
    acc = acc / len(target)
    recall = recall / len(target)
    return torch.tensor(acc).cuda().float(), torch.tensor(recall).cuda().float()
```



3.训练结果

（1）GhostNet

```
Epoch 59:Acc 0.604 Recall 0.501
```

（2）ShuffleNetV2

```
Epoch 59:Acc 0.676 Recall 0.613
```



4.测试

测试代码test_checkpoint-029.py：

```python
# 此脚本用于测试集群上训练好的模型精度和召回率是否都达到80%以上

import torch
import torch.nn as nn
import torchvision
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
from torch.utils.data import DataLoader
#from network import ShuffleNetV1
from network_108 import ShuffleNetV2
import os
import subprocess
import math
import numpy as np

def make_label(label_path):
    count = 0
    #with open(label_path,'r',encoding='utf-8') as f:
    with open(label_path,'r') as f:
        dic = []
        for line in f.readlines():
            line = line.strip('\n') # 去掉换行符
            b=line.split(':') #将每一行以冒号为分隔符转换成列表
            del b[1]
            b.append(count)
            dic.append(b)
            count = count + 1
        dic=dict(dic)
    return dic
def make_dataset(dir, dic):
    images = []
    for line in open(dir):
        line = line.strip('\n')
        b = line.split(' ') # 以空格为分隔符将一行字符转换为列表
        length = len(b) # 获取列表b的长度
        path = b[0] # 获取图片路径
        label_array = np.zeros(108) # 全0的标签数组
        for i in range(length - 2):
            label_array[dic[b[i + 2]]] = 1 #所属的类别置1
        item = (path, label_array)
        images.append(item)
    return images


def accuracy(output, target):
    acc = recall = 0
    pred = torch.sigmoid(output).ge(0.5).cpu()
    pred = [pred_one.nonzero().reshape(-1).numpy().tolist() for pred_one in pred]
    #print(pred)
    target = target.cpu()
    target = [target_one.nonzero().reshape(-1).numpy().tolist() for target_one in target]
    #print(target)
    #print(len(target))
    for i in range(len(target)):
        if len(pred[i]) != 0:
            acc += len(set(pred[i])&set(target[i])) / (len(pred[i])+1e-7)
            recall += len(set(pred[i])&set(target[i])) / (len(target[i])+1e-7)
    acc = acc / len(target)
    recall = recall / len(target)
    return torch.tensor(acc).cuda().float(), torch.tensor(recall).cuda().float()


device = torch.device('cuda')

# 数据预处理
data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),  # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，
    # 转换成形状为[C,H,W]，取值范围是[0,1]的torch.FloadTensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])  # 给定均值：(R,G,B) 方差：>（R，G，B），将会把Tensor正则化。
    # 即：Normalized_image=(image-mean)/std。
])


# 加载模型
print('load ShuffleNetV2 model begin!')
#model = ShuffleNetV1(group=3)
model = ShuffleNetV2(model_size='1.0x')
#model = ShuffleNetV2()
#model = nn.DataParallel(model)
#model = torch.nn.DataParallel(model)
#checkpoint = torch.load('./model_best_0_5.pth.tar')
#model.load_state_dict(checkpoint['state_dict'])
#model.load_state_dict(torch.load('./checkpoint-029.pth'))
model.load_state_dict(torch.load('./checkpoint-029.pth')['model'])
model.eval()  # 固定batchnorm，dropout等，一定要有
model = model.to(device)
print('load ShuffleNetV2 model done!')


torch.no_grad()


label_path = './labelsmap.txt'
test_path = './test.txt'
dic = make_label(label_path)
#print(dic)
images = make_dataset(test_path, dic)
length = len(images)
#print(images[0])
count = 0
acc = 0
recall = 0
print('test start!')
for i in range(length):
    print('count:' + str(count))
    item = images[i]
    path = item[0]
    label = item[1]
    label = label.reshape((1, 108)) # 把标签变成batch_size的形式，即使batch_size为1
    Tensor_label = torch.from_numpy(label) # numpy数组转Tensor
    #print('path:' + path)
    #print('label:' + str(label))
    #print('Tensor_label:' + str(Tensor_label))
    #print(type(path))
    #print(type(label))
    #print(type(Tensor_label))
    try:
        img = Image.open(path)
        img = data_transform(img)
        img = img.unsqueeze(0)
        img_ = img.to(device)
        output = model(img_)
        pred = torch.sigmoid(output).ge(0.5)
        count = count + 1
        print(pred)
        print(Tensor_label)
        acc_tmp, recall_tmp = accuracy(output, Tensor_label)
        print('acc_tmp:' + str(acc_tmp))
        print('recall_tmp:' + str(recall_tmp))
        acc = acc + acc_tmp
        recall = recall + recall_tmp
        print('acc:' + str(acc/count))
        print('recall:' + str(recall/count))
    except Exception as e:
        continue

print('final acc:' + str(acc/count))
print('final recall:' + str(recall/count))
```

此测试代码有两个注意的地方：

1.使用data_transform将图像灰度值转换到[-1,1]之间，代码为：

```python
data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),  # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，
    # 转换成形状为[C,H,W]，取值范围是[0,1]的torch.FloadTensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])  # 给定均值：(R,G,B) 方差：>（R，G，B），将会把Tensor正则化。
    # 即：Normalized_image=(image-mean)/std。
])
```

ToTensor()能够把灰度范围从[0,255]变换到[0,1]之间，而后面的transform.Normalize()则把[0,1]变换到[-1,1]。具体地说，对每个通道而言，Normalize执行操作：img = (img-mean)/std，所以需要将三个通道的均值和标准差都设置为0.5，就可以将图像灰度映射到[-1,1]之间；

2.在加载模型遇到的问题经过调试，正确的加载代码为：

```python
model = ShuffleNetV2(model_size='1.0x')
model.load_state_dict(torch.load('./checkpoint-029.pth')['model'])
```

（1）由于线上训练的模型是用python 2.x版本来训练的，如果测试时使用python3.x的版本来加载测试，则会报错：

UnicodeDecodeError: 'ascii' codec can't decode byte 0x9a in position 0: ordinal not in range(128)

解决方法是更换python 2.x版本进行模型加载和测试。

（2）如果网络定义与模型参数加载对不上，如：

```python
model = ShuffleNetV2()
```

默认为1.5x，但是训练阶段设置的宽度为1.0x，会报错：

RuntimeError: Error(s) in loading state_dict for ShuffleNetV2:**size mismatch** for features.0.branch_main.0.weight:.......

此报错说明二者不匹配（即size mismatch），所以一定要保证网络定义和模型参数加载一致。

（3）模型保存为tar包或者pth文件时，并不一定只有训好的网络参数。一般将每个epoch训完的模型保存为一个字典，每个键对应一个值，所以要搞清楚网络参数对应的那个键是什么，才可以加载出网络参数，如：

```python
model.load_state_dict(torch.load('./checkpoint-029.pth'))
```

此时误以为pth文件只包含网络参数，所以导致报错：

RuntimeError: Error(s) in loading state_dict for ShuffleNetV2:**Missing key(s)** in state_dict: "first_conv.0.weight”,......**Unexpected key(s)** in state_dict: "epoch", "optimizer", "model".

此报错说明缺少相应的键"first_conv.0.weight,...”，并且遇到了多余的键"epoch", "optimizer", "model"。原因在于，加载到的pth文件实际上是一个字典文件，本身存在键"epoch", "optimizer", "model"，这些键对应的值分别为训练的epoch数，优化的方式，以及模型参数。所以正确的加载方式如下：

```python
model.load_state_dict(torch.load('./checkpoint-029.pth')['model'])
```

先加载出键”model”对应的值，即模型参数之后，才可以找到网络对应的键"first_conv.0.weight,...”



测试结果：

```
......
acc_tmp:tensor(1.0000, device='cuda:0')
recall_tmp:tensor(1.0000, device='cuda:0')
acc:tensor(0.5755, device='cuda:0')
recall:tensor(0.4472, device='cuda:0')
count:98179
tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
       device='cuda:0', dtype=torch.uint8)
tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       dtype=torch.float64)
acc_tmp:tensor(1.0000, device='cuda:0')
recall_tmp:tensor(1.0000, device='cuda:0')
acc:tensor(0.5755, device='cuda:0')
recall:tensor(0.4472, device='cuda:0')
final acc:tensor(0.5755, device='cuda:0')
final recall:tensor(0.4472, device='cuda:0')
```

结果显示，精度和召回率并没有达到80%以上，需要进一步研究原因。

仔细研究后发现，是精度和召回率的测试函数计算方法有一些问题，分为四种情况：

（1）模型预测的结果和标签都有1；

（2）模型预测的结果有1，标签没有1；

（3）标签有1，模型预测的结果没有1；

（4）模型预测结果和标签都没有1；

前三种情况，用当前的计算方法没有太大问题，第四种情况精度和召回率理论上应该计算为1，但是按照当前的方法计算为0，当测试集中的数据如果很多标签没有1的话，那么会出现较大误差，更改准确率和召回率计算函数如下，在accuracy2的基础之上增加了第四种预测情况：

```python
def accuracy3(output, target):
    acc = recall = 0
    pred = torch.sigmoid(output).ge(0.5)
    print('pred:' + str(pred))
    print('pred[1].nonzero():' + str(pred[1].nonzero()))
    print('pred[1].nonzero().reshape():' + str(pred[1].nonzero().reshape(-1)))
    pred = [pred_one.nonzero().reshape(-1).numpy().tolist() for pred_one in pred]
    target = [target_one.nonzero().reshape(-1).numpy().tolist() for target_one in target]
    print('pred:' + str(pred))
    print('target:' + str(target))
    for i in range(len(target)):
        #print(pred[i])
        #print(target[i])
        # 如果模型预测和标签都没有1，则精度和召回率均视为1
        if (len(pred[i]) == 0)&(len(target[i]) == 0):
            acc += 1.0
            recall += 1.0
        else:
            #print('set:' + str(set(pred[i])&set(target[i])))
            acc += len(set(pred[i])&set(target[i])) / (len(pred[i]) + 1e-7)
            recall += len(set(pred[i])&set(target[i])) / (len(target[i])+1e-7)
    acc = acc / len(target)
    recall = recall / len(target)
    return torch.tensor(acc).float(), torch.tensor(recall).float()
```

使用更改之后的精度召回率测量函数再次测试集群上训好的模型以及单机上训好的模型，结果为：

集群上训好的ShuffleNetV2模型结果（精度约为90%，召回率接近80%）：

```
......
count:98179
tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
       device='cuda:0', dtype=torch.uint8)
tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       dtype=torch.float64)
acc_tmp:tensor(1.0000, device='cuda:0')
recall_tmp:tensor(1.0000, device='cuda:0')
acc:tensor(0.9017, device='cuda:0')
recall:tensor(0.7733, device='cuda:0')
final acc:tensor(0.9017, device='cuda:0')
final recall:tensor(0.7733, device='cuda:0')
```

单机上训好的ShuffleNetV2模型结果（测试结果比之前的测试结果也高了很多，都在70%以上）：

```
......
count:98184
tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0')
tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       dtype=torch.float64)
acc_tmp:tensor(1.0000, device='cuda:0')
recall_tmp:tensor(1.0000, device='cuda:0')
acc:tensor(0.7989, device='cuda:0')
recall:tensor(0.7318, device='cuda:0')
final acc:tensor(0.7989, device='cuda:0')
final recall:tensor(0.7318, device='cuda:0')
```



5.使用训好的模型测试一批业务数据，输出每张图像的标签，存为txt文件

测试代码：test_business_data.py：

```python
#!coding=utf-8
# 此脚本使用训好的宽度为1.0的ShuffleNetV2模型测试一批业务数据，并将识别结果存为txt文件

import torch
import torch.nn as nn
import torchvision
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
from torch.utils.data import DataLoader
#from network import ShuffleNetV1
from network_108 import ShuffleNetV2
import os
import subprocess
import math
import numpy as np

def make_label(label_path):
    count = 0
    #with open(label_path,'r',encoding='utf-8') as f:
    with open(label_path,'r') as f:
        dic = []
        for line in f.readlines():
            line = line.strip('\n') # 去掉换行符
            b=line.split(':') #将每一行以冒号为分隔符转换成列表
            del b[1]
            b.append(count)
            tmp = b[1]
            b[1] = b[0]
            b[0] = tmp
            dic.append(b)
            count = count + 1
        dic=dict(dic)
    return dic

device = torch.device('cuda')

# 数据预处理
data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),  # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，
    # 转换成形状为[C,H,W]，取值范围是[0,1]的torch.FloadTensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])  # 给定均值：(R,G,B) 方差：>（R，G，B），将会把Tensor正则化。
    # 即：Normalized_image=(image-mean)/std。
])


# 加载模型
print('load ShuffleNetV2 model begin!')
#model = ShuffleNetV1(group=3)
model = ShuffleNetV2(model_size='1.0x')
#model = ShuffleNetV2()
#model = nn.DataParallel(model)
#model = torch.nn.DataParallel(model)
#checkpoint = torch.load('./model_best_0_5.pth.tar')
#model.load_state_dict(checkpoint['state_dict'])
#model.load_state_dict(torch.load('./checkpoint-029.pth'))
model.load_state_dict(torch.load('./checkpoint-029.pth')['model'])
model.eval()  # 固定batchnorm，dropout等，一定要有
model = model.to(device)
print('load ShuffleNetV2 model done!')

torch.no_grad()

label_path = './labelsmap.txt' # 标签索引和对应的英文名
business_label_result = './business_label_result.txt' # 业务数据标签存放处
image_path = '/mnt/data1/gyy/multilabel/mobilenet/images/' # 业务数据文件夹
dic = make_label(label_path) # 字典dic键为类别索引，值为对应的英文名称
#print(dic[1])
#print(type(dic[1]))

print('test start!')
count = 0 # 31685张正常图片（可以测试的图片），共33345张图片
with open(business_label_result, 'w') as f: # 每次重新打开都要清空里面的内容，所以打开之后一次性把内容写完
    for jpg in os.listdir(image_path):
        path = image_path + jpg
        try:
            img = Image.open(path)
            img = data_transform(img)
            img = img.unsqueeze(0)
            img_= img.to(device)
            output = model(img_)
            pred = torch.sigmoid(output).ge(0.5)
            #print('pred:' + str(pred))
            #print('pred[0]:' + str(pred[0]))
            count = count + 1
            print('current jpg:' + str(count))
            f.write(jpg)
            for i in range(len(pred[0])):
                if pred[0][i]==1:
                    f.write(' ' + dic[i])
            f.write('\n') # 标签信息写完换行
        except Exception as e:
            continue
```

得到业务数据的识别结果，存为txt文件，如下：

```
01D26516-2A84-7393-9258-DA86E2CC6AD320190907.jpg
77A32284-D9E7-BD54-684C-C8C991D1422C20191018.jpg
9BD457F8-48C8-F2B6-7E2E-48F0EDFDE97620190418.jpg chatscreen
D25DE0D6-AF63-8520-4032-40F01BAE5E1420181102.jpg person vehicle_interior
41680CAD-C86C-5452-41BA-58B51CE1AA5C20190219.jpg
A810ECEC-D38B-938C-0143-C99EBF1655F420190811.jpg person
425ED3F6-4FB9-6E06-1BA6-72FEFCD8365220181224.jpg
679FE138-4696-E300-A1DB-BD1B1DA8743C20181229.jpg outdoor sculpt
610F23EB-0F92-BC0C-38ED-C4CAF651CEAF20190702.jpg
1C07E4EC-8A37-7788-ED71-EBA43E22367F20190729.jpg outdoor
791D99A4-0B51-DF7E-8025-5E1C46DC18B420190915.jpg
AA5C5BEE-E00F-10CD-EBD6-77AC0F7D3C3E20191006.jpg flowers
DAE792E6-566F-C81F-C744-81F4666247A020190401.jpg
3540F04D-4FC3-DDE9-15D0-7F60296078DA20190330.jpg person
A5C73999-24B1-62B9-0CB5-957914866A2E20190206.jpg person
D6818799-9438-C7A9-D6C9-1D72842E0A1C20190421.jpg clouds sky water outdoor scenery
95B1CF44-44C8-7311-9E95-E4F6140BC7AD20190130.jpg
0F3BDE08-AF34-9915-0BE0-C073E03FEC9C20181031.jpg
2FA21F6A-C99C-80C4-36D9-9DBF6415853720190910.jpg
CBB95E9C-4CE2-5F4B-C876-22809C66FE3F20190715.jpg outdoor scenery
......
```



6.业务数据测试结果分析

从txt的结果中可以看出很多图像并没有被打上标签，有些图像被打上标签可能未必正确，需要进行分析，分析如下问题：

1）未被打上标签的图像有哪些？未被打上标签的图像是因为不属于指定的类别，还是类别特征不明显导致没有打上相应的标签？

2）观察打上标签的图像都是什么标签？统计各个标签的数量，就可以看出这批业务数据大致是什么类型的数据。选取数量最高的10-20个标签类别，在有标签的图像中，随机选取100-200张图像，依次根据这些标签统计每个类别的准确率，精度和召回率。

（1）划分打上标签和没有打上标签的图像，并统计这批业务数据中数量最多前10个标签split_statistic.py

```python
# 划分有标签和没有标签的业务图像至不同的文件夹，并在有标签的图像当中统计各个类别的数量

import subprocess

label_path = './labelsmap.txt'
business_data_result = './business_label_result.txt'

# 获取标签字典，键为类别名称，值为类别名称出现的次数，所有类别次数均初始化为0
def make_label(label_path):
    with open(label_path,'r') as f:
        dic = []
        for line in f.readlines():
            line = line.strip('\n') # 去掉换行符
            b=line.split(':') #将每一行以冒号为分隔符转换成列表
            del b[1]
            b.append(0)
            dic.append(b)
        dic=dict(dic)
    return dic

def main():
    dic = make_label(label_path)
    #print(dic)
    # 划分有标签和无标签的图像，并统计标签数量
    with open(business_data_result,'r') as f:
        for line in f.readlines():
            line = line.strip('\n') # 去掉换行符
            b=line.split(' ') #将每一行以冒号为分隔符转换成列表
            if len(b)==1:
                cmd = 'cp ./images/' + b[0] + ' ' + './image_without_label/'
            else:
                cmd = 'cp ./images/' + b[0] + ' ' + './image_with_label/'
                for i in range(1, len(b)):
                    dic[b[i]] += 1 # 相应的类别数加1
            subprocess.call(cmd, shell=True)
            #print(b[0])
    print(dic)


if __name__=='__main__':
    main()
```

运行结果：

划分了有标签的业务图像16110张和无标签业务图像15577张，并确定了各个标签类别出现的次数：

```
{'airport': 4, 'animal': 725, 'beach': 14, 'bear': 6, 'birds': 40, 'boats': 24, 'condom': 8, 'bridge': 18, 'buildings': 696, 'cars': 102, 'castle': 0, 'cat': 85, 'cityscape': 22, 'clouds': 1425, 'coral': 29, 'cow': 0, 'dog': 101, 'fire': 1, 'fish': 27, 'flowers': 481, 'food': 589, 'fox': 0, 'garden': 16, 'glacier': 4, 'grass': 80, 'harbor': 0, 'horses': 0, 'house': 27, 'lake': 251, 'leaf': 66, 'moon': 25, 'mountain': 130, 'nighttime': 455, 'ocean': 113, 'person': 8997, 'plane': 14, 'plants': 492, 'railroad': 6, 'rainbow': 4, 'road': 214, 'rocks': 37, 'sand': 0, 'sky': 2137, 'snow': 109, 'soccer': 9, 'sports': 46, 'street': 39, 'sunset': 18, 'sun': 15, 'surf': 0, 'swimmers': 0, 'tattoo': 1, 'temple': 0, 'tiger': 2, 'tower': 31, 'town': 0, 'train': 4, 'tree': 290, 'valley': 64, 'vehicle': 327, 'water': 883, 'wedding': 1, 'whales': 0, 'selfie': 4315, 'indoor': 921, 'outdoor': 5223, 'bus': 1, 'bicycle': 4, 'motorcycle': 47, 'basketball': 1, 'pool': 6, 'volleyball': 0, 'amusement': 25, 'waterpark': 7, 'danceroom': 12, 'bedroom': 112, 'gym': 47, 'chatscreen': 234, 'chatscreen_momo': 42, 'chatscreen_wechat': 26, 'game': 17, 'animation': 210, 'scenery': 1761, 'delicacy': 241, 'cooking': 0, 'rabbit': 1, 'guina_pig': 0, 'electronic_device': 90, 'cellphone': 28, 'monitor': 6, 'computer': 21, 'nightclub': 28, 'drink': 121, 'wine': 53, 'tea': 23, 'shopping_mall': 11, 'station': 5, 'dining_room': 32, 'vehicle_interior': 146, 'badmington_court': 0, 'square': 0, 'car_show': 35, 'car_key': 0, 'painting': 0, 'art_exhibition': 3, 'sculpt': 39, 'villa': 8, 'selfie_mobile': 100}
```



（2）分析测试结果

1）分析无标签图像的特点

无标签的业务图像和有标签的业务图像各占约50%，分别为15576张，16109张。无标签的业务图像中约有40%文字或图案类的图像，这些图像与108个类别没有关系，故没有打上标签；剩下的和108个类别相关，但是也没有被打上标签，这些在每个类别当中会具体分析原因（体现在召回率低）。



2）分析有标签图像中，数量最多的前11个标签各个类别的准确率，精度和召回率

top-1:'person',8997
top-2:'outdoor',5223
top-3:'selfie',4315
top-4:'sky',2137
top-5:'scenery',1761
top-6:'clouds',1425
top-7:'indoor',921
top-8:'water',883
top-9:'animal',725
top-10:'building',696
top-11:'food',589

每个类别随机选择100张图像（有标签和无标签各随机选择50张）来统计精度和召回率acc,recall：

```
1.person:
真值为person的有41张；
预测为person的有25张；
真值为person，且预测为person的有23张；
精度为：23/25=92%
召回率为：23/41=56.1%

```

```
2.outdoor:
真值为outdoor的有22张；
预测为outdoor的有14张；
真值为outdoor，且预测为outdoor的有13张；
精度为：13/14=92.9%
召回率为：13/22=59.1%
```

```
3.selfie:
真值为selfie的有16张；
预测为selfie的有13张；
真值为selfie，且预测为selfie的有12张；
精度为：12/13=92.3%
召回率为：12/16=75%
```



以下的标签由于出现的次数很少，决定每次随机取200张图像（有标签和无标签各随机选择100张）进行统计各项指标：

```
4.sky:
真值为sky的有32张；
预测为sky的有11张；
真值为sky，且预测为sky的有11张；
精度为：11/11=100%
召回率为：11/32=34.4%
```

```
5.scenery:
真值为scenery的有28张；
预测为scenery的有7张；
真值为scenery，且预测为scenery的有7张；
精度为：7/7/=100%
召回率为：7/28=25%
```

```
6.clouds:
真值为clouds的有8张；
预测为clouds的有5张；
真值为clouds，且预测为clouds的有3张；
精度为：3/5=60%
召回率为：3/8=37.5%
```

```
7.indoor:
真值为indoor的有32张；
预测为indoor的有10张；
真值为indoor，且预测为indoor的有7张；
精度为：7/10=70%
召回率为：7/32=21.9%
```

```
8.water:
真值为water的有12张；
预测为water的有8张；
真值为water，且预测为water的有5张；
精度为：5/8=62.5%
召回率为：5/12=41.7%
```



以下的标签由于出现的次数更少，决定每次随机取400张图像（有标签和无标签各随机选择200张）进行统计各项指标：

```
9.animal:
真值为animal的有11张；
预测为animal的有5张；
真值为animal，且预测为animal的有4张；
精度为：4/5=80%
召回率为：4/11=36.4%
```

```
10.building:
真值为building的有28张；
预测为building的有10张；
真值为building，且预测为building的有10张；
精度为：10/10=100%
召回率为：10/28=35.7%
```

```
11.food:
真值为food的有47张；
预测为food的有5张；
真值为food，且预测为food的有5张；
精度为：5/5=100%
召回率为：5/47=10.6%
```





