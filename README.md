### Requirements
- PyTorch 1.3
- CUDA 10.2

### Launching the code

1. First download the data with ``download_data.sh /path/to/save/data/``. 
2. Please copy ``decathlon_mean_std.pickle`` to the data folder. 
3. Train base model(ResNet26) from scratch on CIFAR-100, then adapting to 9 targets
    ```
        python train_val_all.py
    ```

4. 10 different domains

![domain_samples](https://github.com/JiaMingLin/residual_adapters/blob/master/files/domain_samples.png?raw=true)
