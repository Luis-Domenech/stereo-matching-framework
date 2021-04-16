import numpy as np
import cv2
import torch
from sklearn.feature_extraction import image as sk

from aanet.dataloader import transforms


# Normalizes a regular rgb image by dividing by 255
def normalize(image):
    if isinstance(image, list):
        image = np.asarray(image)


    image_normed = image.astype("float32")
    image_normed /= 255

    return image_normed


# Receives image and applies z score normalization
def zScoreNormalize(image):
    if isinstance(image, list):
        image = np.asarray(image)

    image_normed = image.astype("float32")
    mean, std = cv2.meanStdDev(image_normed)
    image_normed = (image_normed - mean) / std

    return image_normed

# Takes a normalized float image and returns it to normal
# Technically, we are re normalizing it, but deniormalizing makes more sense in this context
# Works regardless of grayscale or rgb input
def deNormalize(image):
    if isinstance(image, list):
        image = np.asarray(image)

    image_normed = 255 * (image - image.min()) / (image.max() - image.min())
    image_normed = np.array(image_normed, np.intc)

    return image_normed

def extractPatches(image, shape, offset = (0,0), stride = (1,1)):
    """Extracts (typically) overlapping regular patches from a grayscale image

    Changing the offset and stride parameters will result in images
    reconstructed by reconstruct_from_grayscale_patches having different
    dimensions! Callers should pad and unpad as necessary!

    Args:
        image (HxW ndarray): input image from which to extract patches

        shape (2-element arraylike): shape of that patches as (h,w)

        offset (2-element arraylike): offset of the initial point as (y,x)

        stride (2-element arraylike): vertical and horizontal strides

    Returns:
        patches (ndarray): output image patches as (N,shape[0],shape[1]) array

        origin (2-tuple): array of top and array of left coordinates
    """

    px, py = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    l, t = np.meshgrid(
        np.arange(offset[1], image.shape[1] - shape[1] + 1, stride[1]),
        np.arange(offset[0], image.shape[0] - shape[0] + 1, stride[0]) )
    
    l = l.ravel()
    t = t.ravel()

    x = np.tile(px[None,:,:], (t.size,1,1)) + np.tile(l[:,None,None], (1,shape[0],shape[1]))
    y = np.tile(py[None,:,:], (t.size,1,1)) + np.tile(t[:,None,None], (1,shape[0],shape[1]))

    return image[y.ravel(),x.ravel()].reshape((t.size,shape[0],shape[1])), (t, l)



def aanetPreprocess(left, right):
    # AANet passes images in rgb format
    left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)

    # Used for transform tensors geenerate below
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Converts images to torch tensors
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    sample = dict()

    sample["left"] = left
    sample["right"] = right

    sample = transform(sample)

    return sample

    

def net1Preprocess(image, patches = None, resizeDims = None):
    if isinstance(image, list):
        image = np.asarray(image)

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize if needed
    if resizeDims is not None:
        image = cv2.resize(image, resizeDims, interpolation = cv2.INTER_AREA)

    # Extract patches
    if patches is not None:
        patches, _ = extractPatches(image, shape = patches, stride = patches)
        # patches = sk.extract_patches_2d(image, patches)
        return zScoreNormalize(patches)

    else:
        return zScoreNormalize(image)



# Function to load a pytorch model
def loadPyTorchModel(net, pretrained_path, return_epoch_iter=False, resume=False, no_strict=False):
    if pretrained_path is not None:
        if torch.cuda.is_available():
            state = torch.load(pretrained_path, map_location='cuda')
        else:
            state = torch.load(pretrained_path, map_location='cpu')

        from collections import OrderedDict
        new_state_dict = OrderedDict()

        weights = state['state_dict'] if 'state_dict' in state.keys() else state

        for k, v in weights.items():
            name = k[7:] if 'module' in k and not resume else k
            new_state_dict[name] = v

        if no_strict:
            net.load_state_dict(new_state_dict, strict=False)  # ignore intermediate output
        else:
            net.load_state_dict(new_state_dict)  # optimizer has no argument `strict`

        if return_epoch_iter:
            epoch = state['epoch'] if 'epoch' in state.keys() else None
            num_iter = state['num_iter'] if 'num_iter' in state.keys() else None
            best_epe = state['best_epe'] if 'best_epe' in state.keys() else None
            best_epoch = state['best_epoch'] if 'best_epoch' in state.keys() else None
            return epoch, num_iter, best_epe, best_epoch