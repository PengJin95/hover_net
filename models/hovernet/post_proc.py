import cv2
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
from skimage.segmentation import watershed
import torch.nn.functional as F

# From: https://github.com/Dana-Farber-AIOS/pathml/blob/master/pathml/ml/hovernet.py
def remove_small_objs(array_in, min_size):
    """
    Removes small foreground regions from binary array, leaving only the contiguous regions which are above
    the size threshold. Pixels in regions below the size threshold are zeroed out.

    Args:
        array_in (np.ndarray): Input array. Must be binary array with dtype=np.uint8.
        min_size (int): Minimum size of each region.

    Returns:
        np.ndarray: Array of labels for regions above the threshold. Each separate contiguous region is labelled with
            a different integer from 1 to n, where n is the number of total distinct contiguous regions
    """
    assert (
        array_in.dtype == np.uint8
    ), f"Input dtype is {array_in.dtype}. Must be np.uint8"
    # remove elements below size threshold
    # each contiguous nucleus region gets a unique id
    n_labels, labels = cv2.connectedComponents(array_in)
    # each integer is a different nucleus, so bincount gives nucleus sizes
    sizes = np.bincount(labels.flatten())
    for nucleus_ix, size_ix in zip(range(n_labels), sizes):
        if size_ix < min_size:
            # below size threshold - set all to zero
            labels[labels == nucleus_ix] = 0
    return labels


def _post_process_single_hovernet(
    np_out, hv_out, small_obj_size_thresh=10, kernel_size=21, h=0.5, k=0.5
):
    """
    Combine predictions of np channel and hv channel to create final predictions.
    Works by creating energy landscape from gradients, and the applying watershed segmentation.
    This function works on a single image and is wrapped in ``post_process_batch_hovernet()`` to apply across a batch.
    See: Section B of HoVer-Net article and
    https://github.com/vqdang/hover_net/blob/14c5996fa61ede4691e87905775e8f4243da6a62/models/hovernet/post_proc.py#L27

    Args:
        np_out (torch.Tensor): Output of NP branch. Tensor of shape (2, H, W) of logit predictions for binary classification
        hv_out (torch.Tensor): Output of HV branch. Tensor of shape (2, H, W) of predictions for horizontal/vertical maps
        small_obj_size_thresh (int): Minimum number of pixels in regions. Defaults to 10.
        kernel_size (int): Width of Sobel kernel used to compute horizontal and vertical gradients.
        h (float): hyperparameter for thresholding nucleus probabilities. Defaults to 0.5.
        k (float): hyperparameter for thresholding energy landscape to create markers for watershed
            segmentation. Defaults to 0.5.
    """
    # compute pixel probabilities from logits, apply threshold, and get into np array
    np_preds = F.softmax(np_out, dim=0)[1, :, :]
    np_preds = np_preds.numpy()

    np_preds[np_preds >= h] = 1
    np_preds[np_preds < h] = 0
    np_preds = np_preds.astype(np.uint8)

    np_preds = remove_small_objs(np_preds, min_size=small_obj_size_thresh)
    # Back to binary. now np_preds corresponds to tau(q, h) from HoVer-Net paper
    np_preds[np_preds > 0] = 1
    tau_q_h = np_preds

    # normalize hv predictions, and compute horizontal and vertical gradients, and normalize again
    hv_out = hv_out.numpy().astype(np.float32)
    h_out = hv_out[0, ...]
    v_out = hv_out[1, ...]
    # https://stackoverflow.com/a/39037135
    h_normed = cv2.normalize(
        h_out, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    v_normed = cv2.normalize(
        v_out, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    h_grad = cv2.Sobel(h_normed, cv2.CV_64F, dx=1, dy=0, ksize=kernel_size)
    v_grad = cv2.Sobel(v_normed, cv2.CV_64F, dx=0, dy=1, ksize=kernel_size)

    h_grad = cv2.normalize(
        h_grad, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    v_grad = cv2.normalize(
        v_grad, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    # flip the gradient direction so that highest values are steepest gradient
    h_grad = 1 - h_grad
    v_grad = 1 - v_grad

    S_m = np.maximum(h_grad, v_grad)
    S_m[tau_q_h == 0] = 0
    # energy landscape
    # note that the paper says that they use E = (1 - tau(S_m, k)) * tau(q, h)
    # but in the authors' code the actually use: E = (1 - S_m) * tau(q, h)
    # this actually makes more sense because no need to threshold the energy surface
    energy = (1.0 - S_m) * tau_q_h

    # get markers
    # In the paper it says they use M = sigma(tau(q, h) - tau(S_m, k))
    # But it makes more sense to threshold the energy landscape to get the peaks of hills.
    # Also, the fact they used sigma in the paper makes me think that this is what they intended,
    m = np.array(energy >= k, dtype=np.uint8)
    m = binary_fill_holes(m).astype(np.uint8)
    m = remove_small_objs(m, min_size=small_obj_size_thresh)

    # nuclei values form mountains so inverse to get basins for watershed
    energy = -cv2.GaussianBlur(energy, (3, 3), 0)
    out = watershed(image=energy, markers=m, mask=tau_q_h)

    return out


def post_process_batch_hovernet(
    outputs, n_classes, small_obj_size_thresh=10, kernel_size=21, h=0.5, k=0.5
):
    """
    Post-process HoVer-Net outputs to get a final predicted mask.
    See: Section B of HoVer-Net article and
    https://github.com/vqdang/hover_net/blob/14c5996fa61ede4691e87905775e8f4243da6a62/models/hovernet/post_proc.py#L27

    Args:
        outputs (list): Outputs of HoVer-Net model. List of [np_out, hv_out], or [np_out, hv_out, nc_out]
            depending on whether model is predicting classification or not.

            - np_out is a Tensor of shape (B, 2, H, W) of logit predictions for binary classification
            - hv_out is a Tensor of shape (B, 2, H, W) of predictions for horizontal/vertical maps
            - nc_out is a Tensor of shape (B, n_classes, H, W) of logits for classification

        n_classes (int): Number of classes for classification task. If ``None`` then only segmentation is performed.
        small_obj_size_thresh (int): Minimum number of pixels in regions. Defaults to 10.
        kernel_size (int): Width of Sobel kernel used to compute horizontal and vertical gradients.
        h (float): hyperparameter for thresholding nucleus probabilities. Defaults to 0.5.
        k (float): hyperparameter for thresholding energy landscape to create markers for watershed
            segmentation. Defaults to 0.5.

    Returns:
        np.ndarray: If n_classes is None, returns det_out. In classification setting, returns (det_out, class_out).

            - det_out is np.ndarray of shape (B, H, W)
            - class_out is np.ndarray of shape (B, n_classes, H, W)

            Each pixel is labelled from 0 to n, where n is the number of individual nuclei detected. 0 pixels indicate
            background. Pixel values i indicate that the pixel belongs to the ith nucleus.
    """

    assert len(outputs) in {2, 3}, (
        f"outputs has size {len(outputs)}. Must have size 2 (for segmentation) or 3 (for "
        f"classification)"
    )
    if n_classes is None:
        np_out, hv_out = outputs
        # send ouputs to cpu
        np_out = np_out.detach().cpu()
        hv_out = hv_out.detach().cpu()
        classification = False
    else:
        assert len(outputs) == 3, (
            f"n_classes={n_classes} but outputs has {len(outputs)} elements. Expecting a list "
            f"of length 3, one for each of np, hv, and nc branches"
        )
        np_out, hv_out, nc_out = outputs
        # send ouputs to cpu
        np_out = np_out.detach().cpu()
        hv_out = hv_out.detach().cpu()
        nc_out = nc_out.detach().cpu()
        classification = True

    batchsize = hv_out.shape[0]
    # first get the nucleus detection preds
    out_detection_list = []
    for i in range(batchsize):
        preds = _post_process_single_hovernet(
            np_out[i, ...], hv_out[i, ...], small_obj_size_thresh, kernel_size, h, k
        )
        out_detection_list.append(preds)
    out_detection = np.stack(out_detection_list)

    if classification:
        # need to do last step of majority vote
        # get the pixel-level class predictions from the logits
        nc_out_preds = F.softmax(nc_out, dim=1).argmax(dim=1) #BHW

        out_classification = np.zeros_like(nc_out.numpy(), dtype=np.uint8) #BCHW

        for batch_ix, nuc_preds in enumerate(out_detection_list): #HW
            # get labels of nuclei from nucleus detection
            nucleus_labels = list(np.unique(nuc_preds)) #label range
            if 0 in nucleus_labels:
                nucleus_labels.remove(0)  # 0 is background
            nucleus_class_preds = nc_out_preds[batch_ix, ...] #HW, class_id

            out_class_preds_single = out_classification[batch_ix, ...] #CHW

            # for each nucleus, get the class predictions for the pixels and take a vote
            for nucleus_ix in nucleus_labels:
                # get mask for the specific nucleus
                ix_mask = nuc_preds == nucleus_ix #HW, t/f cell
                votes = nucleus_class_preds[ix_mask] #HW, 
                majority_class = np.argmax(np.bincount(votes))
                out_class_preds_single[majority_class][ix_mask] = nucleus_ix

            out_classification[batch_ix, ...] = out_class_preds_single

        return out_detection, out_classification
    else:
        return out_detection