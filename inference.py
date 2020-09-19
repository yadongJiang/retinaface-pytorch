from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer




class Inference(object):
    def __init__(self, weight_path, network, use_cpu = False):
        self.weight_path = weight_path
        self.network = network
        self.use_cpu = use_cpu
        self.resize = 1
        self.confidence_threshold = 0.02
        self.nms_threshold = 0.4
        self.vis_thres = 0.5
        self.input_height = 720
        self.input_width = 1280

        self._initialize_weight()

        self.scale = torch.Tensor([1280, 720, 1280, 720]).to(self.device)
        self.prior_data = self._initialize_priorbox(self.cfg, self.input_height, self.input_width)
    
    def _initialize_weight(self):
        self.cfg = None
        if self.network == "mobile0.25":
            self.cfg = cfg_mnet
        elif self.network == "resnet50":
            self.cfg = cfg_re50
        
        self.net = RetinaFace(cfg=self.cfg, phase = 'test')
        self.net = self._load_model(self.net, self.weight_path, self.use_cpu)
        self.net.eval()
        print('Finished loading model!')
        print(self.net)
        cudnn.benchmark = True
        self.device = torch.device("cpu" if self.use_cpu else "cuda")
        print("self. device : ", self.device)
        self.net = self.net.to(self.device)
    
    def _initialize_priorbox(self, cfg, im_height, im_width):
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        
        return prior_data
    
    def _remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def _check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print('Missing keys:{}'.format(len(missing_keys)))
        print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True
    
    def _load_model(self, model, pretrained_path, load_to_cpu):
        print('Loading pretrained model from {}'.format(pretrained_path))
        if load_to_cpu:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self._remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self._remove_prefix(pretrained_dict, 'module.')
        self._check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model
    
    def _forward(self, img_raw):
        # img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_raw is None:
            print("img is None")
            return None, None, None

        img = np.float32(img_raw)
        if self.resize != 1:
            img = cv2.resize(img, None, None, fx=self.resize, fy=self.resize, interpolation=cv2.INTER_LINEAR)
        
        # im_height, im_width, _ = img.shape
        # scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        # scale = scale.to(self.device)


        loc, conf, landms = self.net(img)  # forward pass

        # start = time.time()
        # prior_data = self._initialize_priorbox(self.cfg, im_height, im_width)
        # print("cost time : ", time.time() - start)

        # decode boxes
        boxes = decode(loc.data.squeeze(0), self.prior_data, self.cfg['variance'])
        boxes = boxes * self.scale / self.resize
        boxes = boxes.cpu().numpy()

        # scores
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # landmarks
        landms = decode_landm(landms.data.squeeze(0), self.prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / self.resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)

        dets = dets[keep, :]
        landms = landms[keep]

        dets = np.concatenate((dets, landms), axis=1)

        boxes_list = []
        socres_list = []
        landmarks_list = []
        for b in dets:
            if b[4] < self.vis_thres:
                continue
            
            s = b[4]
            b = list(map(int, b))
            boxes_list.append([b[0], b[1], b[2], b[3]])
            socres_list.append(s)
            landmarks_list.append([b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14]])
        
        return boxes_list, socres_list, landmarks_list
    
    def __call__(self, img_raw):
        return self._forward(img_raw)

class faceDetect(object):
    def __init__(self, weight_path, network):
        self.infer = Inference(weight_path, network)
    
    def detect(self, img):
        boxes, scores, landmarks = self.infer(img)
        return boxes, scores, landmarks

    def imshow(self, img, boxes, scores, landmarks):
        for i in range(len(boxes)):
            box = boxes[i]
            score = scores[i]
            landmark = landmarks[i]

            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

            cx = box[0]
            cy = box[1] + 12
            text = "{:4f}".format(score)
            cv2.putText(img, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            
            # landms
            cv2.circle(img, (landmark[0], landmark[1]), 1, (0, 0, 255), 4)
            cv2.circle(img, (landmark[2], landmark[3]), 1, (0, 255, 255), 4)
            cv2.circle(img, (landmark[4], landmark[5]), 1, (255, 0, 255), 4)
            cv2.circle(img, (landmark[6], landmark[7]), 1, (0, 255, 0), 4)
            cv2.circle(img, (landmark[8], landmark[9]), 1, (255, 0, 0), 4)
        
        cv2.imshow("img", img)
        cv2.waitKey()

face_detect = faceDetect('./weights/mobilenet0.25_Final.pth', 'mobile0.25')
# 人脸、人脸关键点检测入口函数
def faceDetectAPI(img):
    new_width = 1280
    new_height = 720
    img = cv2.resize(img, (new_width, new_height))

    boxes, scores, landmarks = face_detect.detect(img)  ## for return
    face_detect.imshow(img, boxes, scores, landmarks)  ## just for imshow

if __name__ == '__main__':
    import os
    import time
    root = '/media/administrator/00006784000048233/300W/afw'
    for im in os.listdir(root):
        img_path = os.path.join(root, im)
        img = cv2.imread(img_path)
        start = time.time()
        faceDetectAPI(img)
        print("cost time : ", time.time() - start)