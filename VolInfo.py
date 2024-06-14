import os
from pathlib import Path
import skimage


class VolInfo(object):
    """
    Container for information about a scroll volume.
    """

    def __init__(self, dir):
        """
        Constructor
        :param dir: The full path of the directory containing the volume's images
        """
        self.dir = dir
        self.name = Path(dir).stem

        img_names = sorted([f for f in os.listdir(dir) if f.endswith('.tif')])
        self.first_img_name = img_names[0]
        self.last_img_name = img_names[-1]
        self.img_name_len = len(Path(img_names[0]).stem)
        self.num_imgs = 1 + int(Path(img_names[-1]).stem) - int(Path(img_names[0]).stem)

        self.img_shape = skimage.io.imread(os.path.join(dir, img_names[0])).shape
        self.img_height = self.img_shape[0]
        self.img_width = self.img_shape[1]
        self.shape = (self.num_imgs, self.img_height, self.img_width)


