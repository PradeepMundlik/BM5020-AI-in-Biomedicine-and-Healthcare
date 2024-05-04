import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt, cm
import torchvision.transforms as T

np.random.seed(0)

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def train_test_split(data, train_ratio, shuffle):
    if shuffle:
        np.random.shuffle(data)
    train_idx = int(len(data) * train_ratio)
    train_set = data[:train_idx]
    test_set = data[train_idx:]
    return train_set, test_set

def batch_data(data, batch_size):
    data = np.array(data)
    batches = []
    for i in range(0, len(data), batch_size):
        batches.append(data[i:i+batch_size])
    return batches

def files_to_tensors(files, im_dir, data_dict):
    tensors, labels = [], []
    for file in files:
        image = cv2.imread(f"{im_dir}/{file}")
        image = preprocess_image(image)
        tensors.append(image)
        im_labels = data_dict[file]
        labels.append(im_labels)
    tensors = np.array(tensors)
    labels = np.array(labels)
    return torch.tensor(tensors), torch.tensor(labels)

# define the preprocessing function
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float32)
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = normalize(image / 255)
    return image

def postprocess_output(output, num_classes):
    output = output.detach().numpy().squeeze()
    segmentation_map = np.argmax(output, axis=0)
    segmentation_map = cm.tab20(segmentation_map.astype(float) / num_classes)
    return segmentation_map

def visualize_results(image, masks, input_boxes=None):
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(image)
    ax[0].set_title('Input Image')
    ax[0].axis('off')
    
    ax[1].set_title('Annoted Image')
    ax[1].imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), ax[1], random_color=True)
    if input_boxes is not None:
        for box in input_boxes:
            show_box(+box.cpu().numpy(), plt.gca())
    plt.axis('off')
    plt.show()

def show_mask(mask, ax, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def pixel_accuracy(predicted_labels, true_labels):
    predicted_labels = np.array(predicted_labels)
    true_labels = np.array(true_labels)
    correct_labels = (predicted_labels == true_labels).sum()
    accuracy = correct_labels / true_labels.size
    return accuracy

def iou_score(predicted_labels, true_labels):
    predicted_labels = np.array(predicted_labels)
    true_labels = np.array(true_labels)
    intersections = (predicted_labels & true_labels).sum(axis=(1, 2))
    unions = (predicted_labels | true_labels).sum(axis=(1, 2))
    iou_scores = intersections / unions
    iou_scores = np.nan_to_num(iou_scores)
    return iou_scores.mean()

def dice_score(predicted_labels, true_labels):
    predicted_labels = np.array(predicted_labels)
    true_labels = np.array(true_labels)
    intersections = (predicted_labels & true_labels).sum(axis=(1, 2))
    denominator = predicted_labels.sum(axis=(1, 2)) + true_labels.sum(axis=(1, 2))
    dice_scores = intersections / denominator
    dice_scores = np.nan_to_num(dice_scores)
    return dice_scores.mean()

def f1_score(predicted_labels, true_labels):
    predicted_labels = np.array(predicted_labels)
    true_labels = np.array(true_labels)
    intersections = (predicted_labels & true_labels).sum(axis=(1, 2))
    precisions = intersections / predicted_labels.sum(axis=(1, 2))
    recalls = intersections / true_labels.sum(axis=(1, 2))
    f1_scores = 2 * precisions * recalls / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores)
    return f1_scores.mean()
