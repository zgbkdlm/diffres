# Image experiments

To run the image experiments, first run the following command in the terminal. 

```bash
mkdir -p datasets
cd datasets

pip install mnists==0.4.1 datasets==4.8.4 dm-pix==0.4.4

# for evaluation only
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torchmetrics==1.9.0
```