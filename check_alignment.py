import os
import sys
from numbers import Number
import pprint
import argparse
import json
import numpy as np
import skimage
from PIL import Image as PILImage
from PIL import ImageOps as PILImageOps
from VolInfo import VolInfo
from scroll_align import  extract_axial_img_from_volume, extract_img_from_volume



def create_overlap_image(source_dir, target_dir, transform, image_geom, binarize):

    source_vol_info = VolInfo(source_dir)
    target_vol_info = VolInfo(target_dir)

    if isinstance(image_geom, Number):
        slice_num = image_geom
        img_path = os.path.join(source_vol_info.dir, str(slice_num).zfill(source_vol_info.img_name_len) + '.tif')
        img_a = (skimage.io.imread(img_path) / 256).astype(np.uint8)
        img_a = np.array(PILImageOps.autocontrast(PILImage.fromarray(img_a), 1, 0))
        img_b = extract_axial_img_from_volume(
            slice_num, source_vol_info, target_vol_info, transform, img_cache=None, eight_bit=True)
    else:
        ul, ur, ll = image_geom
        img_a = extract_img_from_volume(source_vol_info, ul, ll, ur, np.identity(4), img_cache=None, eight_bit=True)
        img_b = extract_img_from_volume(target_vol_info, ul, ll, ur, transform, img_cache=None, eight_bit=True)

    if binarize:
        img_a = np.array(img_a > skimage.filters.threshold_otsu(img_a), dtype=np.uint8)
        img_b = np.array(img_b > skimage.filters.threshold_otsu(img_b), dtype=np.uint8)

    rgb_img = np.stack([img_b, img_a, np.zeros(img_a.shape, dtype=np.uint8)], axis=2)
    return rgb_img



if __name__ == '__main__':

    # Build a parser for the command-line args
    parser = argparse.ArgumentParser(prog='Check Alignment', description='Creates before-and-after alignment images.')
    parser.add_argument('-sd', '--source_dir', help='The source-volume directory')
    parser.add_argument('-td', '--target_dir', help='The target-volume directory')
    parser.add_argument('-bx', '--before_xfrm', help='The json file containing the "before" transform')
    parser.add_argument('-ax', '--after_xfrm', help='The json file containing the "after" transform')
    parser.add_argument('-sn', '--slice_num', type=int, default=None, help='The slice number to align')
    parser.add_argument('-sc', '--slice_corners', nargs=9, type=float, default=None, help=
        'The upper-left, upper-right, and lower-left corners of the alignment image. Takes precedence over slice_num')
    parser.add_argument('-bn', '--binarize', default=False, action=argparse.BooleanOptionalAction, help='Binarize the alignment images')
    parser.add_argument('-o', '--out_file', help='The path of the output file')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    print('\n Check Alignment running with these params: \n')
    pprint.pp(vars(args), compact=True, width=120)

    # Process the args
    source_dir = args.source_dir
    target_dir = args.target_dir

    with open(args.before_xfrm) as xfrm_file:
        before_xfrm = np.array(json.load(xfrm_file)['params'])
    with open(args.after_xfrm) as xfrm_file:
        after_xfrm = np.array(json.load(xfrm_file)['params'])

    image_geom = args.slice_num if args.slice_num is not None else \
        [[args.slice_corners[3*i + j] for j in range(3)] for i in range(3)]

    # Create and save the output images
    before = PILImage.fromarray( create_overlap_image(source_dir, target_dir, before_xfrm, image_geom, args.binarize) )
    after = PILImage.fromarray( create_overlap_image(source_dir, target_dir, after_xfrm, image_geom, args.binarize) )

    if args.binarize:
        palette = [255,255,0,  0,255,0,  255,0,0,  0,0,0]
        before.save(args.out_file, format='GIF', palette=palette, save_all=True, append_images=[after], duration=2000, loop=0)
    else:
        before.save(args.out_file, format='GIF', save_all=True, append_images=[after], duration=2000, loop=0)

    print('Done.')