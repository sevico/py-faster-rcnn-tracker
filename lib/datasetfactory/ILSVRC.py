import os
import os.path as osp
import xml.etree.ElementTree as ET

import numpy as np
import scipy.sparse
import scipy.io as sio

# from utils.cython_bbox import bbox_overlaps


years = {'2013': '2013',
         '2014': ['0000', '0001', '0002', '0003', '0004', '0005', '0006']}

name = 'ILSVRC'


def prepare_train_text(dataset):
    """ Prepare training text with xml annotation files.
    """
    # First step: get images' parent directories
    ann_root = osp.join(dataset, 'Annotations')
    _root_2013 = name + years['2013'] + '_train'
    _2013 = osp.join(ann_root, 'DET', 'train', _root_2013)
    dirs_2013 = [osp.join('DET', 'train', _root_2013, dir) for dir in os.listdir(_2013)]
    dirs_2014 = [osp.join('DET', 'train', name + '2014' + '_train_' + sub) for sub in years['2014']]

    dirs = dirs_2013 + dirs_2014

    # Second step: get all the xml file paths
    xmls = []
    for _dir in dirs:
        this_xmls = [osp.join(_dir, xml) for xml in os.listdir(osp.join(ann_root, _dir))]
        xmls += this_xmls

    print 'There are {} xml files.'.format(len(xmls))

    # Third step: parse xml files and assign class labels
    sysnets = open(osp.join(dataset, 'sysnets.txt'), 'wb')
    classes = []
    for xml in xmls:
        filename = osp.join(ann_root, xml)
        tree = ET.parse(filename)
        objs = tree.findall('object')
        for obj in objs:
            objname = obj.find('name').text.strip()
            if objname not in classes:
                classes.append(objname)
    classes.sort()
    # insert __background__
    classes.insert(0, '__background__')
    for ind, _class in enumerate(classes):
        sysnets.write(_class + ' ' + str(ind) + '\n')
    sysnets.close()

    # Fourth step: write train
    train_txt = open(osp.join(dataset, 'train.txt'), 'wb')
    for ix, xml in enumerate(xmls):
        img_path = osp.splitext(xml)[0]
        train_txt.write(img_path + '\n')
        if (ix + 1) % 1000 == 0:
            print 'Processed {} files'.format(ix + 1)
    train_txt.close()


def load_annotation(num_classes, xml, class_indexes):
    tree = ET.parse(xml)
    objs = tree.findall('object')
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, num_classes), dtype=np.float32)
    # "Seg" area for pascal is just the box area
    seg_areas = np.zeros((num_objs), dtype=np.float32)

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1
        cls = class_indexes[obj.find('name').text.lower().strip()]
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes[ix] = cls
        overlaps[ix, cls] = 1.0
        seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

    overlaps = scipy.sparse.csr_matrix(overlaps)

    return {'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}


def _load_data(dataset, class_indexes):
    train_txt = osp.join(dataset, 'train.txt')
    with open(train_txt, 'rb') as f:
        train_datas = [train_data.strip('\n') for train_data in f.readlines()][: 10000]
    image = [osp.join(dataset, 'Data', train_data) + '.JPEG' for train_data in train_datas]
    annotations = [osp.join(dataset, 'Annotations', train_data) + '.xml' for train_data in train_datas]

    roidb = [load_annotation(len(class_indexes), xml, class_indexes) for xml in annotations]
    # add image path to each entry
    for ind, entry in enumerate(roidb):
        entry['image'] = image[ind]
    
    return roidb


def _load_class_labels(dataset):
    sysnets = osp.join(dataset, 'sysnets.txt')
    sysnets = open(sysnets, 'rb')
    clabels = [clabel.strip('\n') for clabel in sysnets.readlines()]
    class_labels = {}
    class_indexes = {}
    for clabel in clabels:
        clabel = clabel.split()
        class_labels[int(clabel[1])] = clabel[0]
        class_indexes[clabel[0]] = int(clabel[1])
    return class_labels, class_indexes


def ILSVRC_handler(dataset):
    dataset = dataset['dataset']
    class_labels, class_indexes = _load_class_labels(dataset)
    roidb = _load_data(dataset, class_indexes)
    return class_labels, roidb

if __name__ == '__main__':
    dataset = osp.join('data', 'ILSVRC2015')
    prepare_train_text(dataset)
