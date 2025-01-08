To train on a single GPU, you only need to run following command. 
```shell
python train_net.py 
```
We use the launch utility `torch.distributed.launch` to launch multiple 
processes for distributed training on multiple gpus. `GPU_NUM` should be
replaced by the number of gpus to use.

```shell
python -m torch.distributed.launch --nproc_per_node=GPU_NUM train_net.py
```

### Inference

To inference on a single GPU, you only need to run following command. 
```shell
python test_net.py 
```
For multiple gpus: 

```shell
python -m torch.distributed.launch --nproc_per_node=GPU_NUM test_net.py
```