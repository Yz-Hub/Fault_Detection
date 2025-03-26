
# LightGEUnet  
LightGEUnet is a lightweight model for seismic data processing and serves as a key model for the doctoral dissertation. This project provides comprehensive tools for training, testing, and visualization, suitable for seismic data analysis and fault detection tasks.

## Environment Setup  
Ensure Conda is installed before proceeding. Create the environment using:  
```bash
conda env create -f environment.yml
```

## Model Training  
To train the model, use:  
```bash
python main.py
```

We provide several sample training datasets in the `datasets/train` folder for demonstration purposes:


## Model Testing  
1. Place prediction data (`.npy` format) in the `datasets/test/seismic` folder.  
2. Prediction results will be saved in the `datasets/test/fault` folder.  

Run the test script:  
```bash
python predict_3d.py
```

## Visualization Tools  
### 3D Visualization  
- **View Seismic Data**  
  ```python
  from utils.result_3D import seismic3D
  seismic3D(seismic_path="your_path.npy")
  ```
- **View Fault Data**  
  ```python
  from utils.result_3D import fault3D
  fault3D(fault_path="your_fault_data.npy")
  ```

### 2D Visualization  
```python
# utils/result_2D.py
# Configuration parameters:
# - type: Data dimension (2D/3D)
# - num: Slice plane selection
# - file: Target data file
```

## File Conversion  

### Conversion Utilities

| File | Description |
|------|-------------|
| `sgy_npy_dat.py` | Convert `.npy` files to `.sgy` format |
| `readsgy.ipynb` | Read `.sgy` data and convert to `.npy` format |

Basic usage:
```bash
python sgy_npy_dat.py input.npy output.sgy
```

## FAQ  
- **NaN Errors**  
  If encountering NaN errors (potential model initialization issues):  
  1. Rerun the program  
  2. Modify random seeds and rerun  
  3. Check model initialization parameters  

- **File Path Errors**  
  Ensure input/output paths are correct and folders are pre-created.  
