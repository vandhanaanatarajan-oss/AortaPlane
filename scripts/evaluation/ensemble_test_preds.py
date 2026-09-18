import pandas as pd
import numpy as np

df = pd.read_csv('results/per_fold_predictions.csv')

def ensemble_patient(group):
    normals = group[['pred_normal_D', 'pred_normal_H', 'pred_normal_W']].values
    centres = group[['pred_centre_D', 'pred_centre_H', 'pred_centre_W']].values

    ref = normals[0]
    aligned = np.array([n if np.dot(n, ref) >= 0 else -n for n in normals])

    mean_normal = aligned.mean(axis=0)
    mean_normal = mean_normal / np.linalg.norm(mean_normal)
    mean_centre = centres.mean(axis=0)

    gt_normal = group[['gt_normal_D', 'gt_normal_H', 'gt_normal_W']].values[0]
    gt_centre = group[['gt_centre_D', 'gt_centre_H', 'gt_centre_W']].values[0]

    return pd.Series({
        'pred_normal_D': mean_normal[0], 'pred_normal_H': mean_normal[1], 'pred_normal_W': mean_normal[2],
        'pred_centre_D': mean_centre[0], 'pred_centre_H': mean_centre[1], 'pred_centre_W': mean_centre[2],
        'gt_normal_D': gt_normal[0], 'gt_normal_H': gt_normal[1], 'gt_normal_W': gt_normal[2],
        'gt_centre_D': gt_centre[0], 'gt_centre_H': gt_centre[1], 'gt_centre_W': gt_centre[2],
    })

ensembled = df.groupby(['patient_id', 'model']).apply(ensemble_patient).reset_index()
ensembled.to_csv('results/ensembled_test_predictions.csv', index=False)
print(ensembled)
