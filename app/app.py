import json
import base64
import requests
import time
import hashlib

from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import yaml

app = Flask(__name__)
# 读取YAML配置文件
with open('config.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# 访问配置项
# 预加载YOLO模型
model = YOLO("../model/page/model_- 8 october 2023 18_18.pt")
OCR_API_URL = "https://open.easst.cn/openapi/rest/common/ocr"
OCR_APP_ID = config['ocr']['app_id']
OCR_APP_SECRET = config['ocr']['app_secret']


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取上传的图像文件
        data = json.loads(request.data)
        # image_file = request.files['image']
        key = data.get('image', '')
        if key != 'ebuilder':
            return jsonify({'status': False, 'code': 401, 'error': '无效的key'})
        base64_image = data.get('image', '')
        confidence = data.get('confidence', 0.4)

        if base64_image:
            image_data = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_data))
        else:
            image = data.get('url', '')
        # 调用YOLO模型进行对象检测
        results = model.predict(source=image, conf=confidence, save=True)

        # 检查是否有检测结果
        if not results:
            return jsonify({'status': True, 'code': 200, 'data': []})  # 如果没有检测到对象，返回一个空列表

        # 返回检测结果
        return jsonify({'status': True, 'code': 200, 'data': json.loads(results[0].tojson())})

    except Exception as e:
        return jsonify({'status': False, 'code': 500, 'error': str(e)})


@app.route('/ocr', methods=['POST'])
def ocr():
    try:
        # 获取上传的图像文件
        data = json.loads(request.data)
        key = data.get('image', '')
        if key != 'ebuilder':
            return jsonify({'status': False, 'code': 401, 'error': '无效的key'})
        base64_image = data.get('image', '')

        # 获取当前时间戳（毫秒数）
        timestamp = str(int(time.time() * 1000))
        # 构建签名
        signature = hashlib.md5((OCR_APP_ID + timestamp + OCR_APP_SECRET).encode()).hexdigest()
        # 设置请求头信息
        headers = {
            "appId": OCR_APP_ID,
            "timestamp": timestamp,
            "sign": signature
        }
        # 构建请求参数
        if base64_image:
            image_data = base64.b64decode(base64_image)
        else:
            response = requests.get(data.get('url', ''))
            if response.status_code == 200:
                image_data = response.content
            else:
                return jsonify({'status': False, 'code': 500, 'error': '图片下载失败'})
        files = {'img': image_data}

        # 调用OCR接口
        results = requests.post(OCR_API_URL, files=files, headers=headers)

        # 检查是否有检测结果
        if results.status_code != 200:
            return jsonify({'status': False, 'code': 500, 'error': str(results.content)})
        # 返回检测结果
        json_result = json.loads(results.content)
        if json_result['status_code'] == 5200:
            json_result['status_code'] = 200
        return jsonify(json_result)

    except Exception as e:
        return jsonify({'status': False, 'code': 500, 'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
