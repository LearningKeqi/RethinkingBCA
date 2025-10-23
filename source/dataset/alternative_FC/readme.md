# Code for Generating Alternative Functional Connectivity

**Main Entrance:**
construct_fc_main.py

**Alternative Features Included:**
- Frequency-based Functional Connectivity (Magnitude-squared Coherence)
- Phase-based Functional Connectivity (Phase-Locking Value, PLV)
- Granger Causality
- Partial Directed Coherence
- Pearson Correlation-based Functional Connectivity


## Usage
Example command for generating alternative node features

```bash
python -u ./construct_fc_main.py --dataset ABIDE --method gc             --start 0 --end 1008 --output_dir ./new_fcs
```


