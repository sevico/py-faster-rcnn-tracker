from PIL import Image

from configuration.config import cfg


class IMDB(object):
    """Image Database Class for image pre-processing."""

    def __init__(self):
        self.original_roidb = []
        self.extend_roidb = []
        self.class_labels = {}
        self._roidb_handle = None
        self.roidb = []

    def append_flipped_image(self):
        """Append flipped image to the Region-of-Interest database."""
        for entry in self.original_roidb:
            size = Image.open(entry).size
            boxes = entry['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = size[0] - oldx2 - 1
            boxes[:, 2] = size[0] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            flipped_entry = {
                'image': entry['image'],
                'boxes': boxes,
                'gt_overlaps': entry['gt_overlaps'],
                'gt_classes': entry['gt_classes'],
                'flipped': True}
            self.extend_roidb.append(flipped_entry)

    def append_transpose_image(self):
        """Append transpose image to the Region-of-Interest database."""
        for entry in self.original_roidb:
            transposed_entry = entry.copy()
            transposed_entry['transposed'] = True
            self.extend_roidb.append(transposed_entry)

    def load_data(self, handle, **kwargs):
        """Get the Region-of-Interest generation function handle."""
        self._roidb_handle = handle

        self.class_labels, self.original_roidb = self._roidb_handle(kwargs)
        if cfg.TRAIN.USE_FLIPPED_IMAGE:
            self.append_flipped_image()
        if cfg.TRAIN.USE_TRANSPOSED_IMAGE:
            self.append_transpose_image()

        self.roidb = self.original_roidb + self.extend_roidb

        return self.roidb
