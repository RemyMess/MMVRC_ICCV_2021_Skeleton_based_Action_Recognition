## Generate preprocessed data for UAV-HUMAN dataset

1. Install joblib for parallel processing (optional)
```python
pip install joblib
```

2. Generate the train_raw_data.npy and train_label.pkl 
```python
python generate_data.py --data_path /path/to/pose_data_root
```

3. Generate the train_process_data.npy and train_process_length.npy
```python
python preprocess.py --data_path /path/to/pose_data_root
```
