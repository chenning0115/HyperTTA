import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split


def load_data(data_path, labels_path):
    # Load the .mat files
    data = sio.loadmat(data_path)['paviaU']  # Replace with the actual variable name in the .mat file
    labels = sio.loadmat(labels_path)['paviaU_gt']  # Replace with the actual variable name in the .mat file
    return data, labels


def split_data(X, y, train_size, sign, output_dir):
    classes = np.unique(y)
    TR = np.zeros_like(y)  # Training labels array
    TE = np.zeros_like(y)  # Testing labels array

    for cls in classes:
        if cls == 0:  # Skip background or undefined class
            continue
        indices = np.where(y == cls)
        cls_indices = np.array(list(zip(indices[0], indices[1])))  # Convert to (row, col) pairs

        # Split the indices into training and testing sets
        train_idx, test_idx = train_test_split(cls_indices, train_size=train_size, random_state=42)

        # Assign training labels
        for idx in train_idx:
            TR[idx[0], idx[1]] = cls

        # Assign testing labels
        for idx in test_idx:
            TE[idx[0], idx[1]] = cls

    # Save the split data
    split_data = {
        'input': X,
        'y': y,
        'TR': TR,  # 2D array with training labels
        'TE': TE  # 2D array with testing labels
    }
    file_name = f"{sign}_{int(train_size * 100)}.mat"
    file_path = f"{output_dir}/{file_name}"
    sio.savemat(file_path, split_data)


# Paths to your data
data_path = '../data/PaviaU/PaviaU.mat'
labels_path = '../data/PaviaU/PaviaU_gt.mat'

# Load the data
X, y = load_data(data_path, labels_path)

# Split the data with 80% for training
split_data(X, y, 0.8, 'PaviaU', '../data/PaviaU/')
