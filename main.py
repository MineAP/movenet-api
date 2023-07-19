import cv2
import numpy as np
import io
from PIL import Image
import base64

import tensorflow as tf
import tensorflow_hub as hub

from flask import Flask, abort, request
from flask_restful import Api, Resource
from flask_cors import CORS
import threading

app: Flask = Flask(__name__)
CORS(app)
api: Api = Api(app)
movenet: any
vs = None

KEYPOINT_THRESHOLD = 0.2

def main():
    # Tensorflow Hubを利用してモデルダウンロード
    model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')

    global movenet
    movenet = model.signatures['serving_default']
    
    # サーバー起動
    start_server_thread()


def run_inference(model, image):
    # 画像の前処理
    input_image = cv2.resize(image, dsize=(256, 256))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = np.expand_dims(input_image, 0)
    input_image = tf.cast(input_image, dtype=tf.int32)

    # 推論実行・結果取得
    outputs = model(input_image)
    keypoints = np.squeeze(outputs['output_0'].numpy())

    image_height, image_width = image.shape[:2]
    keypoints_list, scores_list, bbox_list = [], [], []

    # 検出した人物ごとにキーポイントのフォーマット処理
    for kp in keypoints:
        keypoints = []
        scores = []
        for index in range(17):
            kp_x = int(image_width * kp[index*3+1])
            kp_y = int(image_height * kp[index*3+0])
            score = kp[index*3+2]
            keypoints.append([kp_x, kp_y])
            scores.append(score)
        bbox_ymin = int(image_height * kp[51])
        bbox_xmin = int(image_width * kp[52])
        bbox_ymax = int(image_height * kp[53])
        bbox_xmax = int(image_width * kp[54])
        bbox_score = kp[55]

        keypoints_list.append(keypoints)
        scores_list.append(scores)
        bbox_list.append([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, bbox_score])

    return keypoints_list, scores_list, bbox_list

def render(image, keypoints_list, scores_list, bbox_list):
    render = image.copy()
    for i, (keypoints, scores, bbox) in enumerate(zip(keypoints_list, scores_list, bbox_list)):
        if bbox[4] < 0.2:
            continue

        cv2.rectangle(render, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)

        # 0:nose, 1:left eye, 2:right eye, 3:left ear, 4:right ear, 5:left shoulder, 6:right shoulder, 7:left elbow, 8:right elbow, 9:left wrist, 10:right wrist,
        # 11:left hip, 12:right hip, 13:left knee, 14:right knee, 15:left ankle, 16:right ankle
        # 接続するキーポイントの組
        kp_links = [
            (0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,6),(5,7),(7,9),(6,8),
            (8,10),(11,12),(5,11),(11,13),(13,15),(6,12),(12,14),(14,16)
        ]
        for kp_idx_1, kp_idx_2 in kp_links:
            kp_1 = keypoints[kp_idx_1]
            kp_2 = keypoints[kp_idx_2]
            score_1 = scores[kp_idx_1]
            score_2 = scores[kp_idx_2]
            if score_1 > KEYPOINT_THRESHOLD and score_2 > KEYPOINT_THRESHOLD:
                cv2.line(render, tuple(kp_1), tuple(kp_2), (0,0,255), 2)

        for idx, (keypoint, score) in enumerate(zip(keypoints, scores)):
            if score > KEYPOINT_THRESHOLD:
                cv2.circle(render, tuple(keypoint), 4, (0,0,255), -1)

    return render

class Inference(Resource):
    def post(self):
        print('Inference.post()')

        image_base64 = request.json['image']
        image_byte: bytes = base64.b64decode(image_base64)

        #バイナリーストリーム <- バリナリデータ
        img_binarystream = io.BytesIO(image_byte)
        #PILイメージ <- バイナリーストリーム
        img_pil = Image.open(img_binarystream)
        #numpy配列(RGBA) <- PILイメージ
        image = np.asarray(img_pil)

        # image = np.frombuffer( image_byte, dtype=np.uint8 )

        # 推論実行
        keypoints_list, scores_list, bbox_list = run_inference(movenet, image)

        # 画像レンダリング
        result_image = render(image, keypoints_list, scores_list, bbox_list)

        pil_img: Image = Image.fromarray(result_image)
        buffer = io.BytesIO()
        pil_img.save(buffer, format='jpeg')

        # response = app.make_response(buffer.getvalue())
        # response.headers['content-type'] = 'application/octet-stream'
        # response.headers['Content-Disposition'] = 'attachment; filename=inference.jpg'

        # return response

        data: dict = {
            'image': base64.b64encode(buffer.getvalue()).decode(encoding='utf-8')
        }

        return data

def server_thread():
    print('server_thread()')
    api.add_resource(Inference, '/api/inference/')
    app.run(host='0.0.0.0', port=5000)


def start_server_thread():
    print('start_server_thread()')
    threading.Thread(target=server_thread, args=()).start()

if __name__ == '__main__':
    main()
