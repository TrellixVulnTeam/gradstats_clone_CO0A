# note: base schedule is used from 256 gbs setting - here adascale gbs is 64x8=512
_base_ = [
    '../_base_/models/resnext50_32x4d.py',
    '../_base_/datasets/imagenet_bs64.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]
