# Code for Generating Alternative Node Features

**Main Entrance:**
construct_node_feature_main.py

**Alternative Features Included:**
- Statistical Feature Extracted from BOLD Time-series Signals
- Power Spectral Density (PSD) Feature
- Node Degree Feature (See souce/dataset/process_node_feature.py)
- Eigenvector Feature
- Connection Profile Feature (See souce/dataset/process_node_feature.py)
- BOLD Time-Series Embedding Feature (See souce/dataset/process_node_feature.py)

## Usage
Example command for generating alternative node features

```bash
python -u .e/construct_node_feature_main.py --dataset ABCD --method stat --start 0 --end 7326 --output_dir ./new_node_features --overwrite --normalize --refit_norm
```


