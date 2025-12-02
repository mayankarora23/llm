from enum import Enum
import numpy as np
import torch as torch
import mlx.core as mlxc

import torch.utils.data as td

import time

from Logging import Logging
from Logging import LogLevel

class NumPyWrapperBase:
    def __init__(self):
        Logging.GetInstance().Info("NumPyWrapperBase constructor called")

    def maximum(self, x1, x2):
        return np.maximum(x1, x2)

    def exp(self, x):
        return np.exp(x)

    def copy(self, src, dst):
        dst = np.zeros_like(src)
        np.copyto(dst, src)

    def sum(self, a, axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True): 
        return np.sum(a, axis, dtype, out, keepdims, initial, where)

    def dot(self, a, b):
        return np.dot(a, b)

    def max(self, a, axis=None, out=None, keepdims=False, initial=None, where=True):
        return np.max(a, axis, out, keepdims, initial, where)

    def clip(self, a, a_min, a_max, out=None, **kwargs):
        return np.clip(a, a_min, a_max, out, **kwargs)

    def argmax(self, a, axis=None, out=None):
        return np.argmax(a, axis, out)

    def log(self, a):
        return np.log(a)

    def mean(self, a, axis=None, dtype=None, out=None, keepdims=False):
        return np.mean(a, axis, dtype, out, keepdims)

    def array(self, a, dtype=None, copy=True, order='K', subok=False, ndmin=0):
        return np.array(a, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin)

    def zeros(self, shape, dtype=float, order='C'):
        return np.zeros(shape, dtype, order)

    def ones(self, shape, dtype=None, order='C'):
        return np.ones(shape, dtype, order)

    def where(self, condition, x=None, y=None):
        return np.where(condition, x, y)

    def zeros_like(self, a, dtype=None, order='K', subok=True, shape=None):
        return np.zeros_like(a, dtype, order, subok, shape)

    def sqrt(self, x):
        return np.sqrt(x)

    def linspace(self, start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
        return np.linspace(start, stop, num, endpoint, retstep, dtype, axis)

    def meshgrid(self, *xi):
        return np.meshgrid(*xi)

    def vstack(self, tup):
        return np.vstack(tup)

    def concatenate(self, arrays, axis=0, out=None):
        return np.concatenate(arrays, axis, out)

    def getGenerator(self, device : str=""):
        return None

    def eye(self, n, m=None, k=0, dtype=float, order='C'):
        if dtype is None:
            dtype = float
        if dtype == np.float64:
            dtype = np.float64
        elif dtype == np.float32:
            dtype = np.float32
        elif dtype == np.int64:
            dtype = np.int64
        elif dtype == np.int32:
            dtype = np.int32
        elif dtype == np.int16:
            dtype = np.int16
        elif dtype == np.int8:
            dtype = np.int8
        localA = np.eye(n, M=m, k=k, dtype=dtype, order=order)
        return localA

class NumPyWrapperTorch(NumPyWrapperBase):
    def __init__(self):
        super().__init__()
        Logging.GetInstance().Debug("NumPyWrapperTorch constructor called")

    def dot(self, a, b):
        gpuDevice = torch.device("mps")
        if type(a) != torch.Tensor:
            localA = torch.tensor(a, device=gpuDevice, dtype=torch.float32)
        else:
            localA = a

        if type(b) != torch.Tensor:
            localB = torch.tensor(b, device=gpuDevice, dtype=torch.float32)
        else:
            localB = b

        result = torch.matmul(localA, localB)
        return np.array(result.cpu())

    def array(self, a, dtype=None, copy=True, order='K', subok=False, ndmin=0):
        gpuDevice = torch.device("mps")
        if dtype is None:
            dtype = torch.float32
        if dtype == np.float64:
            dtype = torch.float64
        elif dtype == np.float32:
            dtype = torch.float32
        elif dtype == np.int64:
            dtype = torch.int64
        elif dtype == np.int32:
            dtype = torch.int32
        elif dtype == np.int16:
            dtype = torch.int16
        elif dtype == np.int8:
            dtype = torch.int8
        localA = torch.tensor(a, device=gpuDevice, dtype=dtype)
        return localA

    def getGenerator(self, device : str=""):
        if device == "mps":
            generator = torch.Generator(device="mps")
        else:
            generator = torch.Generator()
        return generator

    def eye(self, n, m=None, k=0, dtype=float, order='C'):
        gpuDevice = torch.device("mps")
        if m is None:
            m = n
        if dtype is None:
            dtype = torch.float32
        if dtype == np.float64:
            dtype = torch.float64
        elif dtype == np.float32:
            dtype = torch.float32
        elif dtype == np.int64:
            dtype = torch.int64
        elif dtype == np.int32:
            dtype = torch.int32
        elif dtype == np.int16:
            dtype = torch.int16
        elif dtype == np.int8:
            dtype = torch.int8

        localA = torch.eye(n)
        Logging.GetInstance().Debug(f"NumPyWrapperTorch.eye: localA = {localA}")
        return localA

    def zeros_like(self, a, dtype=None, order='K', subok=True, shape=None):
        gpuDevice = torch.device("mps")
        if dtype is None:
            dtype = torch.float32
        if dtype == np.float64:
            dtype = torch.float64
        elif dtype == np.float32:
            dtype = torch.float32
        elif dtype == np.int64:
            dtype = torch.int64
        elif dtype == np.int32:
            dtype = torch.int32
        elif dtype == np.int16:
            dtype = torch.int16
        elif dtype == np.int8:
            dtype = torch.int8

        localA = torch.zeros_like(a, device=gpuDevice, dtype=dtype)
        Logging.GetInstance().Debug(f"NumPyWrapperTorch.zeros_like: localA = {localA}")
        return localA

class NumPyWrapperMLX(NumPyWrapperBase):
    def __init__(self):
        super().__init__()
        Logging.GetInstance().Info("NumPyWrapperMLX constructor called")

    def dot(self, a, b):
        mlxc.set_default_device(mlxc.DeviceType.gpu)
        localA = mlxc.array(a)
        localB = mlxc.array(b)
        result = mlxc.matmul(localA, localB)
        mlxc.eval(result)
        return np.array(result)

    def array(self, a, dtype=None, copy=True, order='K', subok=False, ndmin=0):
        mlxc.set_default_device(mlxc.DeviceType.gpu)
        localA = mlxc.array(a)
        return localA

    def eye(self, n, m=None, k=0, dtype=float, order='C'):
        mlxc.set_default_device(mlxc.DeviceType.gpu)
        if m is None:
            m = n
        if dtype is None:
            dtype = mlxc.float32
        if dtype == np.float64:
            dtype = mlxc.float64
        elif dtype == np.float32:
            dtype = mlxc.float32
        elif dtype == np.int64:
            dtype = mlxc.int64
        elif dtype == np.int32:
            dtype = mlxc.int32
        elif dtype == np.int16:
            dtype = mlxc.int16
        elif dtype == np.int8:
            dtype = mlxc.int8
        localA = mlxc.eye(n, dtype=mlxc.float32, stream=mlxc.Device(mlxc.DeviceType.gpu))
        return localA

class DummyDataset:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class DummyDataLoader:
    def __init__(self,
                 dataset,
                 drop_last = None,
                 batch_size=32,
                 shuffle=True,
                 generator=None,
                 num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        for start in range(0, len(self.indices), self.batch_size):
            end = start + self.batch_size
            batch_indices = self.indices[start:end]
            batch = [self.dataset[i] for i in batch_indices]
            data, labels = zip(*batch)
            yield np.stack(data), np.array(labels)

    def __len__(self):
        return len(self.dataset) // self.batch_size


class NumPyType(Enum):
    NUM_PY_TYPE     = 0,
    PYTORCH_TYPE    = 1,
    MLX_TYPE        = 2

__numPyType : NumPyType = NumPyType.NUM_PY_TYPE
__numPyObject : NumPyWrapperBase = None

ndarray = np.ndarray
DataLoader = td.DataLoader
DataSet = td.Dataset

int8 = np.int8
int16 = np.int16
int32 = np.int32
int64 = np.int64
float32 = np.float32
float64 = np.float64

pytorchInt8 = torch.int8
pytorchInt16 = torch.int16
pytorchInt32 = torch.int32
pytorchInt64 = torch.int64
pytorchFloat32 = torch.float32
pytorchFloat64 = torch.float64
pytorchFloat = torch.float

def getType(type):
    if __numPyType == NumPyType.NUM_PY_TYPE:
        return np.dtype(type)
    elif __numPyType == NumPyType.PYTORCH_TYPE:
        if type == np.int8:
            return torch.int8
        elif type == np.int16:
            return torch.int16
        elif type == np.int32:
            return torch.int32
        elif type == np.int64:
            return torch.int64
        elif type == np.float32:
            return torch.float32
        elif type == np.float64:
            return torch.float64

    if __numPyType == NumPyType.MLX_TYPE:
        if type == np.int8:
            return mlxc.int8
        elif type == np.int16:
            return mlxc.int16
        elif type == np.int32:
            return mlxc.int32
        elif type == np.int64:
            return mlxc.int64
        elif type == np.float32:
            return mlxc.float32
        elif type == np.float64:
            return mlxc.float64

def changeArrayType(array, type):
    global __numPyObject
    if __numPyType == NumPyType.NUM_PY_TYPE:
        return np.array(array).astype(getType(type))
    elif __numPyType == NumPyType.PYTORCH_TYPE:
        return torch.tensor(array).to(getType(type))
    elif __numPyType == NumPyType.MLX_TYPE:
        return mlxc.array(array).astype(getType(type))

def setNumPyType(numPyType):
    global __numPyType
    global __numPyObject
    global ndarray
    global mlxc
    global torch
    global np
    global DataLoader
    global DataSet

    __numPyType = numPyType
    if numPyType == NumPyType.PYTORCH_TYPE:
        __numPyObject = NumPyWrapperTorch()
        ndarray = torch.tensor
        DataLoader = td.DataLoader
        DataSet = td.Dataset
        torch.set_default_device(torch.device("mps"))
        torch.set_default_dtype(torch.float32)
    elif numPyType == NumPyType.NUM_PY_TYPE:
        ndarray = np.ndarray
        __numPyObject = NumPyWrapperBase()
        DataLoader = DummyDataLoader
        DataSet = DummyDataset
    elif numPyType == NumPyType.MLX_TYPE:
        ndarray = mlxc.array
        __numPyObject = NumPyWrapperMLX()
        DataLoader = DummyDataLoader
        DataSet = DummyDataset

def getNumPyType():
    global __numPyType
    return __numPyType

def maximum(x1, x2):
    global __numPyObject
    return __numPyObject.maximum(x1, x2)

def exp(x):
    global __numPyObject
    return __numPyObject.exp(x)

def copy(src, dst):
    global __numPyObject
    __numPyObject.copy(src, dst)

def sum(a, axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True): 
    global __numPyObject
    return __numPyObject.sum(a, axis, dtype, out, keepdims, initial, where)

def dot(a, b):
    global __numPyObject
    return __numPyObject.dot(a, b)

def max(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    global __numPyObject
    return __numPyObject.max(a, axis, out, keepdims, initial, where)

def clip(a, a_min, a_max, out=None, **kwargs):
    global __numPyObject
    return __numPyObject.clip(a, a_min, a_max, out, **kwargs)

def argmax(a, axis=None, out=None):
    global __numPyObject

    if type(a) == torch.Tensor:
        a = a.cpu().numpy()

    return np.argmax(a, axis, out)

def log(a):
    global __numPyObject
    return __numPyObject.log(a)

def mean(a, axis=None, dtype=None, out=None, keepdims=False):
    global __numPyObject
    return __numPyObject.mean(a, axis, dtype, out, keepdims)

def array(a, dtype=None, copy=True, order='K', subok=False, ndmin=0):
    global __numPyObject
    return __numPyObject.array(a, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin)

def zeros(shape, dtype=float, order='C'):
    return __numPyObject.zeros(shape, dtype, order)

def ones(shape, dtype=None, order='C'):
    global __numPyObject
    return __numPyObject.ones(shape, dtype, order)

def where(condition, x=None, y=None):
    global __numPyObject
    return __numPyObject.where(condition, x, y)

def zeros_like(a, dtype=None, order='K', subok=True, shape=None):
    global __numPyObject
    return __numPyObject.zeros_like(a, dtype, order, subok, shape)

def sqrt(x):
    global __numPyObject
    return __numPyObject.sqrt(x)

def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    global __numPyObject
    return __numPyObject.linspace(start, stop, num, endpoint, retstep, dtype, axis)

def meshgrid(*xi):
    global __numPyObject
    return __numPyObject.meshgrid(*xi)

def vstack(tup):
    global __numPyObject
    return __numPyObject.vstack(tup)

def concatenate(arrays, axis=0, out=None):
    global __numPyObject
    return __numPyObject.concatenate(arrays, axis, out)

def getGenerator(device : str=""):
    global __numPyObject
    return __numPyObject.getGenerator(device)

def eye(n, m=None, k=0, dtype=float, order='C'):
    global __numPyObject
    return __numPyObject.eye(n, m, k, dtype, order)

class Random:
    def __init__(self, numPytype):
        self.__numPyType = numPytype

    def randn(self, *args):
        if self.__numPyType == NumPyType.NUM_PY_TYPE:
            return np.random.randn(*args).astype(np.float32)
        elif self.__numPyType == NumPyType.PYTORCH_TYPE:
            return torch.randn(*args, device=torch.device("mps"), dtype=torch.float32)
        elif self.__numPyType == NumPyType.MLX_TYPE:
            return mlxc.random.normal([args[0], args[1]], dtype=mlxc.float32)

    def seed(self, seed):
        if self.__numPyType == NumPyType.NUM_PY_TYPE:
            return np.random.seed(seed)
        elif self.__numPyType == NumPyType.PYTORCH_TYPE:
            return torch.manual_seed(seed)
        elif self.__numPyType == NumPyType.MLX_TYPE:
            return mlxc.random.seed(seed)

    def shuffle(self, a, axis=0):
        if self.__numPyType == NumPyType.NUM_PY_TYPE:
            return np.random.shuffle(a, axis)
        elif self.__numPyType == NumPyType.PYTORCH_TYPE:
            return torch.shuffle(a, axis)
        elif self.__numPyType == NumPyType.MLX_TYPE:
            return mlxc.shuffle(a)

    def randint(self, low, high=None, size=None, dtype='l'):
        if self.__numPyType == NumPyType.NUM_PY_TYPE:
            return np.random.randint(low, high, size, dtype)
        elif self.__numPyType == NumPyType.PYTORCH_TYPE:
            return torch.randint(low, high, size, dtype)
        elif self.__numPyType == NumPyType.MLX_TYPE:
            return mlxc.randint(low, high, size, dtype)

    def uniform(self, low=0.0, high=1.0, size=None):
        if self.__numPyType == NumPyType.NUM_PY_TYPE:
            return np.random.uniform(low, high, size)
        elif self.__numPyType == NumPyType.PYTORCH_TYPE:
            return torch.rand(size)
        elif self.__numPyType == NumPyType.MLX_TYPE:
            return mlxc.random.uniform(low, high, size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        if self.__numPyType == NumPyType.NUM_PY_TYPE:
            return np.random.normal(loc, scale, size)
        elif self.__numPyType == NumPyType.PYTORCH_TYPE:
            return torch.normal(loc, scale, size)
        elif self.__numPyType == NumPyType.MLX_TYPE:
            return mlxc.normal(loc, scale, size)

    def choice(self, a, size=None, replace=True, p=None):
        if self.__numPyType == NumPyType.NUM_PY_TYPE:
            return np.random.choice(a, size, replace, p)
        elif self.__numPyType == NumPyType.PYTORCH_TYPE:
            return torch.choice(a, size, replace, p)
        elif self.__numPyType == NumPyType.MLX_TYPE:
            return mlxc.choice(a, size, replace, p)

    def permutation(self, x):
        if self.__numPyType == NumPyType.NUM_PY_TYPE:
            return np.random.permutation(x)
        elif self.__numPyType == NumPyType.PYTORCH_TYPE:
            return torch.randperm(x)
        elif self.__numPyType == NumPyType.MLX_TYPE:
            return mlxc.permutation(x)

class Linalg:
    def __init__(self, numPyType):
        self.__numPyType = numPyType
    def norm(self, x, ord=None, axis=None, keepdims=False):
        if self.__numPyType == NumPyType.NUM_PY_TYPE:
            return np.linalg.norm(x, ord, axis, keepdims)
        elif self.__numPyType == NumPyType.PYTORCH_TYPE:
            return torch.norm(x, ord, axis, keepdims)
        elif self.__numPyType == NumPyType.MLX_TYPE:
            return mlxc.norm(x, ord, axis, keepdims)

    def inv(self, a):
        if self.__numPyType == NumPyType.NUM_PY_TYPE:
            return np.linalg.inv(a)
        elif self.__numPyType == NumPyType.PYTORCH_TYPE:
            return torch.inverse(a)
        elif self.__numPyType == NumPyType.MLX_TYPE:
            localA = mlxc.array(a)
            return mlxc.inv(localA)

    def solve(self, a, b):
        if self.__numPyType == NumPyType.NUM_PY_TYPE:
            return np.linalg.solve(a, b)
        elif self.__numPyType == NumPyType.PYTORCH_TYPE:
            return torch.solve(b, a)

    def eig(self, a):
        if self.__numPyType == NumPyType.NUM_PY_TYPE:
            return np.linalg.eig(a)
        elif self.__numPyType == NumPyType.PYTORCH_TYPE:
            return torch.eig(a)
        elif self.__numPyType == NumPyType.MLX_TYPE:
            localA = mlxc.array(a)
            return mlxc.eig(localA)

    def svd(self, a, full_matrices=True, compute_uv=True):
        if self.__numPyType == NumPyType.NUM_PY_TYPE:
            return np.linalg.svd(a, full_matrices, compute_uv)
        elif self.__numPyType == NumPyType.PYTORCH_TYPE:
            return torch.svd(a, compute_uv)
        elif self.__numPyType == NumPyType.MLX_TYPE:
            localA = mlxc.array(a)
            return mlxc.svd(localA, full_matrices, compute_uv)

    def qr(self, a, mode='reduced'):
        if self.__numPyType == NumPyType.NUM_PY_TYPE:
            return np.linalg.qr(a, mode)
        elif self.__numPyType == NumPyType.PYTORCH_TYPE:
            return torch.qr(a)
        elif self.__numPyType == NumPyType.MLX_TYPE:
            localA = mlxc.array(a)
            return mlxc.qr(localA, mode)

    def cholesky(self, a):
        if self.__numPyType == NumPyType.NUM_PY_TYPE:
            return np.linalg.cholesky(a)
        elif self.__numPyType == NumPyType.PYTORCH_TYPE:
            return torch.cholesky(a)
        elif self.__numPyType == NumPyType.MLX_TYPE:
            localA = mlxc.array(a)
            return mlxc.cholesky(localA)

    def pinv(self, a, rcond=1e-15):
        if self.__numPyType == NumPyType.NUM_PY_TYPE:
            return np.linalg.pinv(a, rcond)
        elif self.__numPyType == NumPyType.PYTORCH_TYPE:
            return torch.pinverse(a)
        elif self.__numPyType == NumPyType.MLX_TYPE:
            localA = mlxc.array(a)
            return mlxc.pinv(localA, rcond)

    def det(self, a):
        if self.__numPyType == NumPyType.NUM_PY_TYPE:
            return np.linalg.det(a)
        elif self.__numPyType:
            return torch.det(a)
        elif self.__numPyType == NumPyType.MLX_TYPE:
            localA = mlxc.array(a)
            return mlxc.det(localA)

random = Random(__numPyType)
linalg = Linalg(__numPyType)