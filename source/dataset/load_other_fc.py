import numpy as np

def load_other_fc(cfg):
    dataset_name = cfg.dataset.name # abcd, hcp, abide, pnc
    fc_type = cfg.fc_type # gc, pdc, phase, frequency, pearson

    
    # transfer to upper case
    dataset_name = dataset_name.upper()

    new_fcs_path = f'./alternate_fc/new_fcs/{dataset_name}/{dataset_name}_{fc_type}.npy'
    new_fcs = np.load(new_fcs_path)

    # preprocess: replace any sample matrices containing NaN/Inf with identity
    if new_fcs.ndim == 3:
        num_nodes = new_fcs.shape[1]
        valid_mask = np.isfinite(new_fcs).all(axis=(1, 2))
        num_replaced = int((~valid_mask).sum())
        if num_replaced > 0:
            identity = np.eye(num_nodes, dtype=new_fcs.dtype)
            new_fcs[~valid_mask] = identity
            print(f'replaced {num_replaced} invalid FC matrices with identity')
    elif new_fcs.ndim == 2:
        # handle edge case where a single matrix is loaded
        if not np.isfinite(new_fcs).all():
            num_nodes = new_fcs.shape[0]
            new_fcs = np.eye(num_nodes, dtype=new_fcs.dtype)
            print('input FC had invalid values; replaced with identity')

    print(f'dataset_name={dataset_name}, fc_type={fc_type}')
    print(f'new_fcs.shape={new_fcs.shape}')  # should be (num_samples, num_nodes, num_nodes)

    if fc_type == 'gc_fromsl_P':
        new_fcs = 1 - new_fcs

    return new_fcs
