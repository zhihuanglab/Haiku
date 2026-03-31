import numpy as np
import matplotlib.pyplot as plt

from PIF.single_channel_featurize import SingleChannelHistogram


def get_ROI_image_feature(ROI, full_img, featurizer=SingleChannelHistogram()):
    ROI_img = full_img[:, ROI['yrange'][0]:ROI['yrange'][1], ROI['xrange'][0]:ROI['xrange'][1]]
    patch_image_feats = [featurizer.featurize(im) for im in ROI_img]
    patch_image_feats = np.concatenate(patch_image_feats, 0)
    return patch_image_feats


def get_ROI_coord_feature(ROI):
    return np.array((ROI['center_x'], ROI['center_y'])).reshape((2,))


def plot_region_with_ROIs(full_img, ROIs, colors=['w'], use_channels=[], size=10):
    if use_channels == []:
        use_channels = sorted(np.random.choice(np.arange(full_img.shape[0]), (3,), replace=False))

    plot_image = full_img[use_channels].transpose((1, 2, 0)).astype(float)
    plot_image = np.clip(plot_image / np.quantile(plot_image, 0.95), 0., 1.)

    plt.figure(figsize=(size, size))
    plt.imshow(plot_image)

    if isinstance(ROIs, dict) and 'patch_name' in ROIs:
        ROIs = [ROIs]
    centers = [(ROI['center_x'], ROI['center_y']) for ROI in ROIs]
    if len(centers) > 0:
        centers = np.stack(centers, 0)
        plt.scatter(centers[:, 0], centers[:, 1], c=colors, s=5)
    return
