import cv2
import time
import argparse
import torch
import numpy as np

from lib.networks.architecture.Rec_model import RecModel
from lib.utils.general import generate_alphabets, strLabelConverter, resize_padding


def infer(image, model, device, alphabets):
    converter = strLabelConverter(alphabets)

    model.eval()
    preds = model(image.to(device))

    _, preds = preds.max(2)
    preds = preds.contiguous().view(-1)
    # print(preds)

    preds_size = torch.IntTensor([preds.size(0)])
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('results: {0}'.format(sim_pred))
    return 0


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='the path to image',
                        default='./data/recognition/images/000003_03_01.jpg'
                        # default='/home/lyb/datasets/OCR/Sythetic_Chinese_Character_Dataset/images/20841500_145593715.jpg'
                        )
    parser.add_argument('--cfg', type=str, default='/home/lyb/ocr/text_det_reg/config/models/Rec_MobileNetV3_LSTM_CTC.yml', help='model.yaml path')
    opt = parser.parse_args()

    device = torch.device('cpu')
    # device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')

    alphabets = generate_alphabets(alphabet_path='/home/lyb/ocr/text_det_reg/config/data/alphabets.txt')
    nc = len(alphabets) + 1

    model = RecModel(opt.cfg, ch=3, nc=nc, training=False).to(device)

    ckpt = '/home/lyb/ocr/text_det_reg/runs/exp0/weights/best_RecModel_MobileNetV3_SequenceDecoder_CTC.pt'
    print('===> loading pretrained model from %s' % ckpt)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    # ------在此输入图片路径-------
    start_time1 = time.time()

    image = cv2.imread(opt.image_path)

    # 不足的，补充白色区域
    image = resize_padding(image, 32, 480)
    image_batch = torch.from_numpy(np.expand_dims(image.transpose(2, 0, 1), axis=0).astype(np.float32) / 255.)

    infer(image_batch, model, device, alphabets)

    finish_time = time.time()
    print('load model time: %g' % (finish_time - start_time1))
    print('elapsed time of recognition: %g' % (finish_time - start_time))
