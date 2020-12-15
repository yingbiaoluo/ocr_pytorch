import json
import time
import base64
import requests


if __name__ == '__main__':
    start_time = time.time()
    image_path = '/Users/biaobiao/PycharmProjects/remote_text_det_rec/data/images_question/train_images/000000_03.jpg'

    with open(image_path, 'rb') as f:
        img_byte = base64.b64encode(f.read())
    img_str = img_byte.decode('ascii')

    data = {'recognize_img': img_str}
    url = 'http://61.160.213.177:10433'

    s = requests.session()
    s.keep_alive = False
    response = s.post(url, data=json.dumps(data))

    result = response.text
    print(result)

    print('elapsed time:', time.time() - start_time)  # 总时间




