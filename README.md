### Requirements
- PyTorch 1.3
- CUDA 10.2

### Launching the code
First download the data with ``download_data.sh /path/to/save/data/``. Please copy ``decathlon_mean_std.pickle`` to the data folder. 

Train base model(ResNet26) from scratch on CIFAR-100, then adapting to 9 targets
```
python train_val_all.py
```

10 different domains
