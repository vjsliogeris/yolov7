
import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized
import cv2
import glob

from scipy.spatial.transform import Rotation as R

N_KPTS = 9
DELTA = 20

points_3D = np.array([[ 0., 0.175525,  0.175525, -0.175525, -0.175525,  0.175525,  0.175525, -0.175525, -0.175525],
                      [ 0., 0.175525,  0.175525,  0.175525,  0.175525, -0.175525, -0.175525, -0.175525, -0.175525],
                      [ 0., 0.0665665,-0.0665665,-0.0665665, 0.0665665, 0.0665665,-0.0665665,-0.0665665, 0.0665665]]).T

camera_matrix = np.array([[575, 0, 416],
                          [0, 575, 416],
                          [0,   0,   1]])

PLOT = True

def euler_to_rotation(theta) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]),  np.cos(theta[0])  ]
                    ])

    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                   1,      0                   ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])

    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),     np.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

def evaluate(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         save_json_kpt=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_txt_tidl=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         wandb_logger=None,		 
		 compute_loss=None,
         half_precision=True,
         is_coco=False,
         opt=None,		 		 
         tidl_load=False,
         dump_img=False,
         kpt_label=False,
         flip_test=False):
    #Literally ripping off test.py
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
        (save_dir / 'labels' if save_txt or save_txt_tidl else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    model.model[-1].flip_test = False
    model.model[-1].flip_index = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml') or data.endswith('coco_kpts.yaml')
        with open(data) as f:
            data = yaml.safe_load(f)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    '''
    data[:,0,:] - target
    data[:,1,:] - predictions
    data[:,:,0] - x
    data[:,:,1] - y
    data[:,:,2] - z
    data[:,:,3] - h
    data[:,:,4] - w
    data[:,:,5] - l
    data[:,:,6] - roll
    data[:,:,7] - pitch
    data[:,:,8] - yaw
    '''

    data = []

    image_files = glob.glob('coco/images_all/*')
    for imfname in tqdm(image_files):
    #for imfname in image_files:
        image_number = imfname.split('.jpg')[0].split('images_all/')[1]
        img = cv2.imread(imfname)
        if PLOT:
            cv2.imshow('amogus', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        #Preprocess
        h0, w0 = img.shape[:2]
        r = 832 / max(h0, w0)  # resize image to img_size
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        if PLOT:
            cv2.imshow('amogus', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        img_new = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img_new = img_new[np.newaxis, ...]

        img_new = torch.from_numpy(img_new.copy())
        #Make to tensor (assume it's from dataloader by now
        img_new = img_new.to(device, non_blocking=True)
        img_new = img_new.half() if half else img_new.float()  # uint8 to fp16/32
        img_new /= 255.0  # 0 - 255 to 0.0 - 1.0
        box_info = pd.read_csv('coco/bboxes_all/boxes_' + image_number + '.csv')

        with torch.no_grad():
            out, train_out = model(img_new, augment=False)  # inference and training outputs
            lb = []
            out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, kpt_label=kpt_label, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'])
            prediction = out[0]
            # 6.....33 is positions and confidences
            for pred in prediction:
                bbox = pred[:4]
                xmin = bbox[0].int().item()
                ymin = bbox[1].int().item()
                xmax = bbox[2].int().item()
                ymax = bbox[3].int().item()
                p1 = (xmin, ymin)
                p2 = (xmax, ymax)
                if PLOT:
                    img = cv2.circle(img, p1, 1, (0, 255, 0), 2)
                    img = cv2.circle(img, p2, 1, (0, 255, 0), 2)
                points_2d = []
                for i in range(N_KPTS):
                    x = pred[6+ 3*i].int().item()
                    y = pred[7+ 3*i].int().item()
                    conf = pred[8+ 3*i].item()
                    #if x < xmax + DELTA and x > xmin-DELTA and y < ymax+DELTA and y>ymin-DELTA:
                    if x < xmax + DELTA and x > xmin-DELTA and y < ymax+DELTA and y>ymin-DELTA and not i == 2:
                        if PLOT:
                            img = cv2.circle(img, (x, y), 1, (int(255*conf), int(255*conf), int(255*conf)), 2)
                            print(i)
                            cv2.imshow('amogus', img)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                        points_2d.append([x, y])
                    else:
                        points_2d.append([0, 0])
                points_2D = np.array(points_2d)
                to_remove = (points_2D == np.array([0, 0])).all(axis=1)
                if np.sum(to_remove) > 3:
                    continue
                points_2d = np.delete(points_2D, to_remove, axis=0).astype(float)
                points_3d = np.delete(points_3D, to_remove, axis=0).astype(float)
                dist_coeffs = np.zeros((4,1))
                success, rotation_vector, translation_vector = cv2.solvePnP(
                        points_3d,
                        points_2d,
                        camera_matrix,
                        dist_coeffs,
                        flags=0)
                rot = R.from_rotvec(rotation_vector.T)
                rotmat = rot.as_matrix()
                rotation_X = euler_to_rotation((np.pi, 0, 0))
                translation_vector = np.matmul(rotation_X, translation_vector)
                rotmat = np.matmul(rotation_X, rotmat)
                rot = R.from_matrix(rotmat)
                euler = rot.as_euler('xyz')
                #print("translation")
                #print(translation_vector)
                #print(euler) #!!!!
                dist_min = np.inf
                closest_index = None
                for index, row in box_info.iterrows():
                    x, y, z = row['x'], row['y'], row['z']
                    dx = x - translation_vector[0]
                    dy = y - translation_vector[1]
                    dz = z - translation_vector[2]
                    dist = dx**2 + dy**2 + dz**2
                    if dist < dist_min:
                        dist_min = dist
                        closest_index = index
                candidate_row = box_info.iloc()[closest_index]
                target = [candidate_row['x'],
						  candidate_row['y'],
						  candidate_row['z'],
						  candidate_row['h'],
						  candidate_row['w'],
						  candidate_row['l'],
						  candidate_row['roll'],
						  candidate_row['pitch'],
						  candidate_row['yaw']]
                prediction = [translation_vector[0].item(),
                              translation_vector[1].item(),
                              translation_vector[2].item(),
                              candidate_row['h'],
                              candidate_row['w'],
                              candidate_row['l'],
                              euler[0,0],
                              euler[0,1],
                              euler[0,2]] 
                data.append([target, prediction])
                if PLOT:
                    print(target)
                    print(prediction)
                    cv2.imshow('amogus', img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            if PLOT:
                cv2.imshow('amogus', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        if PLOT:
            cv2.imshow('amogus', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    ar = np.array(data)
    np.save(open('eval.npy',"wb"), ar)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='eval_orientation.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-txt-tidl', action='store_true', help='save results to *.txt in tidl format')
    parser.add_argument('--tidl-load', action='store_true', help='load thedata from a list specified as in tidl')
    parser.add_argument('--dump-img', action='store_true', help='load thedata from a list specified as in tidl')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--save-json-kpt', action='store_true', help='save a cocoapi-compatible JSON results file for key-points')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--kpt-label', action='store_true', help='Whether kpt-label is enabled or not')
    parser.add_argument('--flip-test', action='store_true', help='Whether to run flip_test or not')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_json_kpt |= opt.data.endswith('coco_kpts.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)

    evaluate(opt.data,
         opt.weights,
         opt.batch_size,
         opt.img_size,
         opt.conf_thres,
         opt.iou_thres,
         opt.save_json,
         opt.save_json_kpt,
         opt.single_cls,
         opt.augment,
         opt.verbose,
         save_txt=opt.save_txt | opt.save_hybrid,
         save_txt_tidl=opt.save_txt_tidl,
         save_hybrid=opt.save_hybrid,
         save_conf=opt.save_conf, 
         opt=opt,
         tidl_load = opt.tidl_load,
         dump_img = opt.dump_img,
         kpt_label = opt.kpt_label,
         flip_test = opt.flip_test,
         )
