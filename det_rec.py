import cv2
import yaml
import time
import json
import base64
from easydict import EasyDict as edict
import numpy as np
import torch

from detect_text.models.model import DetRegModel
from detect_text.utils import scale_img
from lib.networks.architecture.Rec_model import RecModel
from lib.utils.general import generate_alphabets, strLabelConverter, padding_image_batch

# os.environ['CUDA_VISIBLE_DEVICES'] = "3"

device = torch.device('cpu')
alphabets = generate_alphabets(alphabet_path='/home/lyb/ocr/text_det_reg/config/data/alphabets.txt')
nc = len(alphabets) + 1


def initializer(context=None):
    with open('/home/lyb/ocr/text_det_reg/detect_text/config/Det_MobileNetV3Large_fpn.yaml') as f:
        config_det = yaml.load(f, Loader=yaml.FullLoader)
        config_det = edict(config_det)

    with open('/home/lyb/ocr/text_det_reg/config/models/Rec_MobileNetV3_LSTM_CTC.yml') as f:
        config_rec = yaml.load(f, Loader=yaml.FullLoader)
        config_rec = edict(config_rec)

    global model_detection
    model_detection = DetRegModel(model_config=config_det['arch']['args'], device=device).to(device)
    global model_recognition
    model_recognition = RecModel(config_rec, ch=3, nc=nc, training=False)

    # ------detection model load weights-----
    ckpt_det = '/home/lyb/ocr/text_det_reg/detect_text/output/mobilenetv3_large_FPN_model_last_infer.pth'
    model_detection.load_state_dict(torch.load(ckpt_det, map_location=device))
    model_detection.eval()

    # ------recognition model load weights---------
    ckpt_rec = '/home/lyb/ocr/text_det_reg/runs/exp0/weights/best_RecModel_MobileNetV3_SequenceDecoder_CTC.pt'
    model_recognition.load_state_dict(torch.load(ckpt_rec, map_location=device))
    model_recognition.eval()


def infer_detection(img):
    start_time = time.time()

    img_scaled, f_scale = scale_img(img)  # (640, 640, 3)
    img_scaled = img_scaled.transpose((2, 0, 1)).astype(np.float32)
    img_scaled = torch.unsqueeze(torch.from_numpy(img_scaled), 0)

    global model_detection

    socres, classes, pred_boxes = model_detection(x=img_scaled.to(device))
    end_time = time.time()
    print('infer time of detection: %gs' % (end_time - start_time))

    return pred_boxes.cpu().detach().numpy() / f_scale


def get_img_batch(image, pred_boxes):
    # sort pred_boxes by second column
    pred_boxes = pred_boxes[pred_boxes[:, 1].argsort()]

    batch_size = pred_boxes.shape[0]
    cropped_images = []
    for i in range(batch_size):
        pred_box = pred_boxes[i]
        image_crop = image[pred_box[1]:pred_box[3], pred_box[0]:pred_box[2], :]
        cropped_images.append(image_crop)

    cropped_images = padding_image_batch(cropped_images, 32, 480)

    return cropped_images


def infer_recognition(image_batch):
    start_time = time.time()

    converter = strLabelConverter(alphabets)

    global model_recognition
    preds = model_recognition(image_batch.to(device))

    _, preds = preds.max(2)
    preds = preds.contiguous().view(-1)

    preds_size = torch.tensor([preds.size(0)], dtype=torch.int)
    result = converter.decode(preds.data, preds_size.data, raw=False)

    finish_time = time.time()
    print('infer time of recognition: {0}'.format(finish_time - start_time))

    return result


def handler(data):
    img_b64 = json.loads(data.decode('utf-8'))['recognize_img']

    im_bytes = base64.b64decode(img_b64)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

    # 文本检测
    pred_boxes = infer_detection(img=img).astype(np.int)

    cropped_images = get_img_batch(img, pred_boxes)

    # 文本识别
    result = infer_recognition(cropped_images)

    return result


if __name__ == '__main__':
    begin_time = time.time()

    initializer()

    image_path = '/home/lyb/ocr/text_det_reg/data/images_question/train_images/000000_01.jpg'
    img = cv2.imread(image_path)
    print('input image shape:', img.shape)

    # 文本检测
    pred_boxes = infer_detection(img=img).astype(np.int)
    print('pred boxes:', pred_boxes)

    cropped_images = get_img_batch(img, pred_boxes)

    # 文本识别
    result = infer_recognition(cropped_images)
    print('results:', result)

    print('Total time: %gs' % (time.time() - begin_time))
