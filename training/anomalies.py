#  Copyright (C) 2020 Canon Medical Systems Corporation. All rights reserved

import random
import numpy as np
from skimage import draw
from skimage.filters import gaussian
from scipy.ndimage import zoom
from skimage.draw import line_aa


def get_all_anomalies():
    return [draw_circle, draw_anomaly1, draw_anomaly2, draw_anomaly3, draw_anomaly4, draw_anomaly5,
            draw_anomaly6, draw_anomaly7, draw_anomaly8, draw_anomaly9, draw_anomaly10, draw_anomaly11,
            draw_anomaly12, draw_anomaly13]


def draw_circle(image, seed=0, radius=None):
    rng = random.Random(seed)

    # radius = image.shape[0]//15
    if radius is None:
        radius = rng.randint(4, 25)
    else:
        radius = image.shape[0] // 15

    r = max(min(int((rng.random() / 2 + 0.25) * image.shape[0]), image.shape[0] - radius - 1), radius + 1)
    c = max(min(int((rng.random() / 2 + 0.25) * image.shape[1]), image.shape[1] - radius - 1), radius + 1)

    color = rng.choice(np.unique(image.flatten()))
    rr, cc = draw.circle(r, c, radius)

    image = np.copy(image)
    image[rr, cc] = color
    labels = np.zeros_like(image)
    labels[rr, cc] = 1.
    return image, labels


def draw_anomaly1(image, seed=0):
    """
    Draws only on the greyish region in the middle of the brain, not on the wriggly edges.
    Circular in shape more or less.
    """
    grid_x, grid_y = np.meshgrid(np.arange(image.shape[-2]), np.arange(image.shape[-1]))
    mask = image > 0.05 #(image > 0.35) & (image < 0.6)

    np.random.seed(seed)
    c_x = np.random.choice(grid_x[mask])
    np.random.seed(seed)
    c_y = np.random.choice(grid_y[mask])

    radius = np.random.randint(15, 30)

    dist = np.sqrt((grid_x - c_x) ** 2 + (grid_y - c_y) ** 2)

    y_mask = dist <= radius * mask

    prob = (1 - np.clip(dist / radius, 0, 1)) / 3 * 2 + (0.33 * y_mask)

    tumour_mask = np.random.binomial(1, prob) * mask
    tumour_values = gaussian(tumour_mask * 1.0, sigma=2) * 1.5 + 1
    anomalous_image = np.clip(image * tumour_values, 0, 1)

    return anomalous_image, y_mask * 1.0


def circles3_shape_mask(image, seed=0, mask=None):
    grid_x, grid_y = np.meshgrid(np.arange(image.shape[-2]), np.arange(image.shape[-1]))
    if mask is None:
        mask = image > 0.05

    # mask = (image > 0.35) & (image < 0.6)

    np.random.seed(seed)
    c_x1 = np.random.choice(grid_x[mask])

    np.random.seed(seed)
    c_y1 = np.random.choice(grid_y[mask])

    radius1 = np.random.randint(9, 30)
    radius2 = np.random.randint(8, radius1)
    radius3 = np.random.randint(8, radius1)

    o_x2 = np.random.randint(-radius1, radius1)
    o_y2 = ((np.random.random() < 0.5) * 2 - 1) * (-1) * np.sqrt(radius1 ** 2 - (o_x2) ** 2)

    o_x3 = np.random.randint(-radius1, radius1)
    o_y3 = ((np.random.random() < 0.5) * 2 - 1) * (-1) * np.sqrt(radius1 ** 2 - (o_x3) ** 2)

    c_x2 = c_x1 + o_x2
    c_y2 = c_y1 + o_y2

    c_x3 = c_x1 + o_x3
    c_y3 = c_y1 + o_y3

    dist1 = np.sqrt((grid_x - c_x1) ** 2 + (grid_y - c_y1) ** 2)
    dist2 = np.sqrt((grid_x - c_x2) ** 2 + (grid_y - c_y2) ** 2)
    dist3 = np.sqrt((grid_x - c_x3) ** 2 + (grid_y - c_y3) ** 2)

    dist = (dist1 < radius1) | (dist2 < radius2) | (dist3 < radius3)

    return dist


def draw_anomaly2(image, seed=0):
    # Replace mask with noise

    mask = circles3_shape_mask(image, seed=seed)

    foreground_mask = image > 0.05
    mean = image[foreground_mask].mean()
    std = image[foreground_mask].std()

    noise = np.random.normal(np.ones_like(image) * mean, np.ones_like(image) * std)

    corrupted_image = np.where(mask, noise, image)
    return corrupted_image, mask * 1.0


def draw_anomaly3(image, seed=0):
    # Replace mask with blurred noise

    mask = circles3_shape_mask(image, seed=seed)

    foreground_mask = image > 0.05
    mean = image[foreground_mask].mean()
    std = image[foreground_mask].std()

    noise = np.random.normal(np.ones_like(image) * mean, np.ones_like(image) * std)
    noise = gaussian(noise, sigma=1)

    corrupted_image = np.where(mask, noise, image)
    return corrupted_image, mask * 1.0


def draw_anomaly4(image, seed=0):
    # Replace mask with blurred original

    mask = circles3_shape_mask(image, seed=seed)

    foreground_mask = image > 0.05
    mean = image[foreground_mask].mean()
    std = image[foreground_mask].std()

    noise = image
    noise = gaussian(noise, sigma=2)

    corrupted_image = np.where(mask, noise, image)
    return corrupted_image, mask * 1.0


def draw_anomaly5(image, seed=0):
    # Replace mask with blurred noise interpolating at the edges

    mask = circles3_shape_mask(image, seed=seed)

    foreground_mask = image > 0.05
    mean = image[foreground_mask].mean()
    std = image[foreground_mask].std()

    noise = np.random.normal(np.ones_like(image) * mean, np.ones_like(image) * std)
    noise = gaussian(noise, sigma=1.5)

    blurry_mask = gaussian(mask * 1.0, sigma=4)

    corrupted_image = image * (1 - blurry_mask) + noise * blurry_mask
    mask = blurry_mask > 0.7

    return corrupted_image, mask * 1.0


def draw_anomaly6(image, seed=0):
    # Circularish shape with the original image with increased intensity

    mask = circles3_shape_mask(image, seed=seed)

    blurry_mask = gaussian(mask * 1.0, sigma=4)

    noise = np.clip(image * 1.3, 0, 1)

    corrupted_image = image * (1 - blurry_mask) + noise * blurry_mask

    mask = blurry_mask > 0.7

    return corrupted_image, mask * 1.0


def draw_anomaly7(image, seed=0):
    # Circularish shape with the original image with decreased intensity

    mask = circles3_shape_mask(image, seed=seed)

    blurry_mask = gaussian(mask * 1.0, sigma=4)

    noise = np.clip(image * 0.7, 0, 1)

    corrupted_image = image * (1 - blurry_mask) + noise * blurry_mask

    mask = blurry_mask > 0.7

    return corrupted_image, mask * 1.0


def draw_anomaly8(image, seed=0):
    # Circularish shape filled with blurred brighter noise interpolating at the edges

    mask = circles3_shape_mask(image, seed=seed)

    foreground_mask = image > 0.05
    mean = image[foreground_mask].mean() + 0.1
    std = image[foreground_mask].std()

    noise = np.random.normal(np.ones_like(image) * mean, np.ones_like(image) * std)
    noise = gaussian(noise, sigma=1.5)

    blurry_mask = gaussian(mask * 1.0, sigma=4)

    corrupted_image = image * (1 - blurry_mask) + noise * blurry_mask

    mask = blurry_mask > 0.7

    return corrupted_image, mask * 1.0


def draw_anomaly9(image, seed=0):
    # Circularish shape filled with shifted pixels interpolating at the edges

    mask = circles3_shape_mask(image, seed=seed)

    rng = random.Random(seed)

    shift_x = rng.randint(20, 40) * (-1 * int(rng.random() > 0.5))
    shift_y = rng.randint(20, 40) * (-1 * int(rng.random() > 0.5))

    noise = np.roll(image, [shift_x, shift_y], axis=[0, 1])

    blurry_mask = gaussian(mask * 1.0, sigma=4)

    corrupted_image = image * (1 - blurry_mask) + noise * blurry_mask

    mask = blurry_mask > 0.7

    return corrupted_image, mask * 1.0


def draw_anomaly10(image, seed=0):
    # Circularish shape filled with blurred hierarchical noise interpolating at the edges

    mask = circles3_shape_mask(image, seed=seed)

    foreground_mask = image > 0.05
    mean = image[mask].mean()
    std = image[mask].std()

    noise = np.zeros_like(image)

    for i in range(1, 8):
        ns = np.random.normal(np.zeros((2 ** i, 2 ** i)), 1)
        ns = zoom(ns, 2 ** (8 - i), order=2)
        noise += ns

    # noise = np.random.normal(np.ones_like(image)*mean, np.ones_like(image)*std)
    noise = gaussian(noise, sigma=1.0)
    noise -= noise.mean()
    noise /= noise.std()
    noise *= std
    noise += mean

    blurry_mask = gaussian(mask * 1.0, sigma=4)

    corrupted_image = image * (1 - blurry_mask) + noise * blurry_mask

    mask = blurry_mask > 0.7

    return corrupted_image, mask * 1.0


def draw_anomaly11(image, seed=0):
    # Circularish shape filled with blurred brighter noise interpolating at the edges

    mask = circles3_shape_mask(image, seed=seed)

    foreground_mask = image > 0.05
    rng = random.Random(seed)

    def get_line():
        def sample():
            r1 = int(rng.random() * image.shape[0])
            c1 = int(rng.random() * image.shape[1])
            r2 = int(rng.random() * image.shape[0])
            c2 = int(rng.random() * image.shape[1])
            return r1, c1, r2, c2

        while True:
            r1, c1, r2, c2 = sample()
            if (r1 - r2) ** 2 + (c1 - c2) ** 2 > 25:
                break

        rr, cc, val = line_aa(r1, c1, r2, c2)

        line_mask = np.zeros_like(image)
        line_mask[rr, cc] = val
        line_mask = (gaussian(line_mask, sigma=3) > 0.01)
        return line_mask

    while True:
        line_mask = get_line()
        if line_mask.sum() > 5:
            break

    blurry_mask = gaussian(line_mask * 1.0, sigma=4)
    mask = blurry_mask > 0.7

    mean = image[mask].mean()
    std = image[mask].std()

    choice = rng.choice(range(3))
    if choice == 0:
        fill = image * 1.5
    elif choice == 1:
        fill = np.random.normal(np.ones_like(image) * mean, np.ones_like(image) * std)
        fill = gaussian(fill, sigma=2)
    elif choice == 2:
        fill = gaussian(image, sigma=4)

    corrupted_image = np.where(line_mask, fill, image)
    corrupted_image = image * (1 - blurry_mask) + corrupted_image * blurry_mask

    return corrupted_image, mask * 1.0


def draw_anomaly12(image, seed=0):
    # Circularish shape filled with nonlinear intensity change interpolating at the edges

    mask = circles3_shape_mask(image, seed=seed)

    foreground_mask = image > 0.05

    x_points = np.linspace(0, 1, 7, endpoint=True)
    y_points = x_points + np.random.normal(loc=0, scale=0.2, size=x_points.shape)

    corrupted = np.interp(image, x_points, y_points)

    blurry_mask = gaussian(mask * 1.0, sigma=4)

    corrupted_image = image * (1 - blurry_mask) + corrupted * blurry_mask

    mask = blurry_mask > 0.7

    return corrupted_image, mask * 1.0


def draw_anomaly13(image, seed=0):

    mask = circles3_shape_mask(image, seed=seed)

    rng = random.Random(seed)

    noise = image.copy()
    m = noise[mask].max()
    if m > 0.2:
        noise /= noise[mask].max() * 0.8 - noise[mask].min()
    else:
        noise = rng.random() * 0.6 + 0.2

    blurry_mask = gaussian(mask * 1.0, sigma=4)

    corrupted_image = image * (1 - blurry_mask) + noise * blurry_mask

    mask = blurry_mask > 0.7

    return corrupted_image, mask * 1.0