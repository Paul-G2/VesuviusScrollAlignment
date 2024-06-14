import os
import sys
import datetime
import json
import argparse
import pprint
import math
import numpy as np
import skimage
from PIL import Image as PILImage
from PIL import ImageOps as PILImageOps
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator
from VolInfo import VolInfo
from ImgCache import ImgCache


def do_alignment(source_dir, target_dir, source_slicenums, initial_xfrm, xfrm_ranges,
                 downsamp_schedule, source_blur=0, target_blur=0):
    """
    Aligns two scroll volumes, by finding an affine transform
    that maximizes their mutual information.

    :param source_dir: the directory containing the source images
    :param target_dir: the directory containing the target images
    :param source_slicenums: the source slices to align
    :param initial_xfrm: an initial guess for the optimal transform
    :param xfrm_ranges: ranges for the transform elements
    :param downsamp_schedule: the sequence of down-sampling factors to apply
    :param source_blur: a gaussian-blur sigma value for the source volume
    :param target_blur: a gaussian-blur sigma value for the target volume
    """

    img_cache = ImgCache()
    source_vol_info = VolInfo(source_dir)
    target_vol_info = VolInfo(target_dir)

    # Get the initial transform and the ranges for each of its elements
    xfrm = np.array(initial_xfrm).reshape((4,4))
    xfrm_ranges = np.array(xfrm_ranges).reshape((3,4))
    xfrm[:3, 3] /= 100 # Scale the shifts so that all params are of order 1
    xfrm_ranges[:3, 3] /= 100
    xfrm_bounds = [(xfrm[i][j] - xfrm_ranges[i][j], xfrm[i][j] + xfrm_ranges[i][j]) for i in range(3) for j in range(4)]

    # Print the initial alignment score
    print('\nCalculating the initial alignment score ...')
    initial_score = alignmentScore(xfrm.flatten()[:12],
        *(source_vol_info, target_vol_info, source_slicenums, img_cache, 1, source_blur, target_blur), initial=True)
    print('Initial alignment score is ' + str(-initial_score) + '\n')

    # Optimize the mutual information at successively higher resolutions
    for dsamp in downsamp_schedule:
        print("\n\nDownsample = " + str(dsamp) + ".  Starting at " +
              datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("------------------------------------------------\n")

        fn_args = (source_vol_info, target_vol_info, source_slicenums, img_cache, dsamp, source_blur, target_blur)

        opt_result = minimize(
            fun      = alignmentScore,
            x0       = xfrm.flatten()[:12],
            args     = fn_args,
            method   = 'Powell',
            bounds   = xfrm_bounds
        )

        # Print results
        global best_mi, best_transform
        print('\nBest result with downsample = ' + str(dsamp) + ':')
        print(str(best_mi))
        print(np.array2string(best_transform, precision=9, separator=', ', max_line_width=100))
        print('\n\n')

        # Prepare to try again with a smaller downsample factor
        xfrm = np.copy(best_transform)
        xfrm_ranges[:3, 3] = 2*dsamp
        xfrm[:3, 3] /= 100
        xfrm_ranges[:3, 3] /= 100
        xfrm_bounds = [(xfrm[i][j] - xfrm_ranges[i][j], xfrm[i][j] + xfrm_ranges[i][j]) \
                       for i in range(3) for j in range(4)]
        best_mi = 0
        best_transform = np.identity(4)
        img_cache.clear()

    print ('Done.  ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def alignmentScore(x, *args, initial=False):
    """
    Calculates the alignment score (ie, the mutual information) for the current transform matrix.
    :param x: the elements of the transform matrix
    :param args: a tuple containing parameters needed by the score function
    """

    (source_vol_info, target_vol_info, source_slicenums, img_cache, dsamp, source_blur, target_blur) = args
    xfrm = np.vstack([np.array(x).reshape((3,4)), [0,0,0,1]])
    xfrm[:3,3] *= 100

    # Get the average mutual information of the (source, target) image pairs
    total_mi = 0
    for slice_num in source_slicenums:
        source_img = get_img(slice_num, source_vol_info, img_cache, eight_bit=True, downsamp=dsamp,
                             blur=source_blur/dsamp)
        target_img = extract_axial_img_from_volume(slice_num, source_vol_info, target_vol_info, xfrm, img_cache,
                                                   eight_bit=True, downsamp=dsamp, target_blur=target_blur/dsamp)
        total_mi += mutual_information(source_img, target_img)
    mi = total_mi/len(source_slicenums)

    # If we got a new best score, then print it
    if initial:
        img_cache.clear()
    else :
        global best_mi, best_transform
        if mi > best_mi:
            best_mi = mi
            best_transform = xfrm
            print('Score: ' + str(best_mi) + ' (downsamp = ' + str(dsamp) + ')')
            print(np.array2string(best_transform, precision=9, separator=', ', max_line_width=100))
            print(' ')

    return -mi


def get_img(slice_num, vol_info, cache, eight_bit, downsamp=1, blur=0):
    """
    Gets an image from the cache if possible, otherwise from disk.
    If loaded from disk, the image is optionally downsampled
    and/or smoothed and/or converted to 8-bits.

    :param slice_num: the slice number of the required image
    :param vol_info: attributes of the source volume
    :param cache: an image cache (may be None)
    :param eight_bit: whether to convert the image to eight bits
    :param downsamp: the down-sampling factor for images read from disk
    :param blur: the amount of blurring to perform
    """

    # Try the cache
    img_path = os.path.join(vol_info.dir, str(slice_num).zfill(vol_info.img_name_len) + '.tif')
    img = cache.get_image(img_path) if cache is not None else None

    # Load from disk if necessary
    if img is None:
        img = skimage.io.imread(img_path)

        if downsamp > 1:
            img = skimage.transform.rescale(img, 1/downsamp, preserve_range=True, anti_aliasing=True)
        if blur >= 0.5:
            img = skimage.filters.gaussian(img, blur, preserve_range=True)
        if eight_bit:
            img = (img / 256).astype(np.uint8)
            img = np.array(PILImageOps.autocontrast(PILImage.fromarray(img), 1, 0)) # Threshold at 1%
        else:
            img = img.astype(np.uint16)

        if cache is not None:
            cache.add_image(img, img_path)

    return img


def extract_axial_img_from_volume(slice_num, source_vol_info, target_vol_info, xfrm, img_cache,
                                  eight_bit, downsamp=1, target_blur=0):
    """
    This is a convenience wrapper for the extract_img_from_volume() function,
    for cases where we just want a full-size axial image

    :param slice_num: the axial slice number in the source volume
    :param source_vol_info: attributes of the source volume
    :param target_vol_info: attributes of the target volume
    :param xfrm: the source-to-target transform
    :param img_cache: an image cache (may be None)
    :param eight_bit: whether to convert the image to eight bits
    :param downsamp: the downsampling factor to apply
    :param target_blur: a gaussian-blur sigma value for the target volume
    """
    h, w = source_vol_info.img_shape
    ul = [0, 0, slice_num + 0.5]
    ll = [0, h, slice_num + 0.5]
    ur = [w, 0, slice_num + 0.5]
    return extract_img_from_volume(target_vol_info, ul, ll, ur, xfrm, img_cache, eight_bit, downsamp, target_blur)


def extract_img_from_volume(target_vol_info, ul, ll, ur, xfrm, img_cache=None, eight_bit=False, downsamp=1, blur=0):
    """
    Samples a transformed volume onto a 2D image plane specified by its corner points.
    :param target_vol_info: attributes of the target volume
    :param ul: upper-left corner
    :param ll: lower-left corner
    :param ur: upper-right corner
    :param xfrm: the source-to-target transform
    :param img_cache: the image cache
    :param eight_bit: whether to convert the image to eight bits
    :param downsamp: the downsampling factor
    """

    # Convert the input points to numpy 1-d arrays
    ul, ur, ll = [np.array(ul), np.array(ur), np.array(ll)]

    # Extend the corners to cover an integer number of pixels
    if downsamp != 1:
        extend_x = math.ceil(np.linalg.norm(ur-ul)) % downsamp
        extend_y = math.ceil(np.linalg.norm(ll-ul)) % downsamp
        ur += extend_x * (ur-ul) / np.linalg.norm(ur-ul)
        ll += extend_y * (ll-ul) / np.linalg.norm(ll-ul)

    # Transform the corner points to the target space, and permute the indices to (z,y,x) order
    corners = [ul, ur, ll, ur+ll-ul]
    dsamp = np.array([1.0, downsamp, downsamp])
    ul_t, ur_t, ll_t, lr_t = corners_t = [np.matmul(xfrm, np.append(c,1))[:3][[2,1,0]]/dsamp for c in corners]

    # Determine the range of coordinate values that we'll need from the target volume
    corner_coords = [[corners_t[n][i] for n in range(4)] for i in range(3)]
    min_z, min_y, min_x = (math.floor(np.min(corner_coords[i])) - 1 for i in range(3))
    max_z, max_y, max_x = (math.ceil(np.max(corner_coords[i])) + 1 for i in range(3))

    # Read the images we need into a 3d array
    dtype = np.uint8 if eight_bit else np.uint16
    img_shape_d = (np.ceil(np.divide(target_vol_info.img_shape,downsamp))).astype(np.int32)
    subvol_d, subvol_h, subvol_w = (max_z - min_z + 1, max_y - min_y + 1, max_x - min_x + 1)
    subvol = np.zeros((subvol_d, subvol_h, subvol_w), dtype=dtype)
    pad_y = (max(0, -min_y), max(0, max_y - img_shape_d[0] + 1))
    pad_x = (max(0, -min_x), max(0, max_x - img_shape_d[1] + 1))

    for z in range(min_z, max_z + 1):
        if z < 0 or z >= target_vol_info.num_imgs:
            subvol_img = np.zeros_like(img_shape_d, dtype=dtype)
        else:
            subvol_img = get_img(z, target_vol_info, img_cache, eight_bit, downsamp, blur)
        subvol_img = np.pad(subvol_img, (pad_y, pad_x))
        subvol[z-min_z, :, :] = subvol_img[min_y+pad_y[0] : max_y+1+pad_y[0], min_x+pad_x[0] : max_x+1+pad_x[0]]

    # Interpolate the sub-volume onto the output-image plane
    subvol_coords = (np.linspace(min_z+0.5, max_z+0.5, subvol_d),
                     np.linspace(min_y+0.5, max_y+0.5, subvol_h), np.linspace(min_x+0.5, max_x+0.5, subvol_w))
    interp = RegularGridInterpolator(subvol_coords, subvol, 'linear', bounds_error=False, fill_value=0)

    out_w = math.ceil(int(np.round(np.linalg.norm(ur-ul)))/downsamp)
    out_h = math.ceil(int(np.round(np.linalg.norm(ll-ul)))/downsamp)
    out_img = np.zeros((out_h, out_w), dtype=dtype)

    row_step = (ur_t - ul_t)/out_w
    col_step = (ll_t - ul_t)/out_h
    sample_coords = np.linspace(ul_t+(row_step+col_step)/2, ur_t-(row_step-col_step)/2, out_w)
    for y in range(out_h):
        out_img[y, :] = np.round(interp(sample_coords)).astype(dtype)
        sample_coords += col_step

    return out_img


def mutual_information(img_a, img_b, border_offset=0.0):
    """
    Computes the mutual information between 2 images.
    :param img_a: the first image
    :param img_b: the second image
    :param border_offset: the fractional size of the border to exclude from the calculation
    """

    h, w = img_a.shape
    if (h,w) != img_b.shape:
        raise Exception("Mismatched image sizes, in mutual_information()")

    f = border_offset
    if f != 0.0:
        img_a = img_a[int(f * h):int((1 - f) * h), int(f * w):int((1 - f) * w)]
        img_b = img_b[int(f * h):int((1 - f) * h), int(f * w):int((1 - f) * w)]

    histo_2d, _, _ = np.histogram2d(img_a.ravel(), img_b.ravel(), bins=256)
    pxy = histo_2d / float(np.sum(histo_2d))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    non_zeros = pxy > 0
    mi = np.sum(pxy[non_zeros] * np.log(pxy[non_zeros] / px_py[non_zeros]))

    return mi


if __name__ == '__main__':

    # Build a parser for the command-line args
    parser = argparse.ArgumentParser(prog='Scroll Align', description='Finds an affine transform that aligns two scroll volumes.')
    parser.add_argument('-v', '--volpkg_dir', help='The volpkg directory')
    parser.add_argument('-s', '--source_vol', help='The name of the source-volume')
    parser.add_argument('-t', '--target_vol', help='The name of the target-volume')
    parser.add_argument('-n', '--slicenums', nargs='+', type=int, help='The list of slice numbers to use')
    parser.add_argument('-d', '--downsamps', nargs='+', type=int, default=1, help='A decreasing sequence of down-sampling factors')
    parser.add_argument('-sb', '--source_blur', type=float, default=0, help='A gaussian-blur sigma value for the source volume')
    parser.add_argument('-tb', '--target_blur', type=float, default=0, help='A gaussian-blur sigma value for the target volume')
    parser.add_argument('-xr', '--xfrm_ranges', nargs=12, type=float, default=None, help='Ranges for the 12 elements of the transform matrix')
    parser.add_argument('-xf', '--initial_xfrm', nargs=16, type=float, default=None, help='The initial transform matrix')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    # Prepare some optimizer params
    source_dir = os.path.join(args.volpkg_dir, 'volumes', args.source_vol)
    target_dir = os.path.join(args.volpkg_dir, 'volumes', args.target_vol)

    if args.initial_xfrm is None:
        initial_xfrm_file = os.path.join(args.volpkg_dir, 'transforms', args.source_vol + '-to-' + args.target_vol + '.json')
        with open(initial_xfrm_file) as xfrm_file:
            args.initial_xfrm = np.array(json.load(xfrm_file)['params'])

    if args.xfrm_ranges is None: # Set reasonable default ranges
        args.xfrm_ranges = [[0.025, 0.02,  0.02,  50],
                            [0.02,  0.025, 0.02,  50],
                            [0.02,  0.02,  0.025, 50]]

    # Do the alignment
    print('\n Scroll Align starting at ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
          ' with these params: \n')
    pprint.pp(vars(args), compact=True, width=120)

    best_mi = 0
    best_transform = np.identity(4)
    do_alignment(source_dir, target_dir, args.slicenums, args.initial_xfrm, args.xfrm_ranges,
                 args.downsamps, args.source_blur, args.target_blur)


