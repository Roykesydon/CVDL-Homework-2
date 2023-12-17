# CVDL-Homework-2

## Environment
- python 3.8

### Setup

1. Install dependencies

   ```shell
   pip install -r requirements.txt --no-cache-dir
   ```

   Windows 要裝有 CUDA 的 Pytorch 再跑以下指令

   ```shell
   pip3 install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

   MacOS 裝 PyQt5 遇到問題的話，建議用 conda 裝
   
   ```shell
   conda install -c anaconda pyqt
   ```

2. Check whether torch is using GPU

    ```shell
    python test_gpu.py
    ```

3. Put pretrained weights `vgg19_bn_993.pth`,`resnet50_with_erase.pth`, `resnet50_without_erase.pth`  in `./weights/`

4. Put `inference_dataset` to `.`

### Run

```
python app.py
```

