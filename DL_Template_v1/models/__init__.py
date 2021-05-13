# -*- utf-8 -*-
# author : fisherwsy

from .ts_causal_learning import LSTM_base,Linear_base,Linear_base_1,Linear_base_3,lstm_base,Dense_base,MLP_base,CNN_base


# 把方法名加到这个包中
__all__ = [
    "LSTM_base",
    "Linear_base",
    "Linear_base_1",
    "Linear_base_3",
    "lstm_base",
    "Dense_base",
    "MLP_base",
    "CNN_base"
]

# fix Sphinx issues, see https://bit.ly/2K2eptM
for item in __all__:
    if hasattr(item, "__module__"):
        setattr(item, "__module__", __name__)