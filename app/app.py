import os
import sys

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取上一级目录的路径
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

sys.path.insert(0, parent_dir)

# 打印sys.path以确认
print(f"sys.path: {sys.path}")
import base64
import hashlib
import inspect
import io
import json
import time

import requests
import yaml
from flask import Flask, jsonify, request
from PIL import Image

from ultralytics import YOLO

print(inspect.getfile(YOLO))
app = Flask(__name__)
# 读取YAML配置文件
with open("config.yml", "r") as config_file:
    config = yaml.safe_load(config_file)

# 访问配置项
# 预加载YOLO模型
model = YOLO("../model/page/model_- 18 december 2023 17_37.pt")
modelV2 = YOLO("../model/page/model_latest.pt")
OCR_API_URL = "https://open.easst.cn/openapi/rest/common/ocr"
OCR_APP_ID = config["ocr"]["app_id"]
OCR_APP_SECRET = config["ocr"]["app_secret"]


@app.route("/predict", methods=["POST"])
def predict():
    try:
        return inner_predict()
    except Exception as e:
        return jsonify({"status": False, "code": 500, "error": str(e)})


@app.route("/predict/v2", methods=["POST"])
def predict_v2():
    try:
        return inner_predict(version=2)
    except Exception as e:
        return jsonify({"status": False, "code": 500, "error": str(e)})


@app.route("/predict/v3", methods=["POST"])
def predict_v3():
    try:
        return inner_predict(version=3)
    except Exception as e:
        return jsonify({"status": False, "code": 500, "error": str(e)})


def inner_predict(version=1):
    # 获取上传的图像文件
    data = json.loads(request.data)
    # image_file = request.files['image']
    key = data.get("key", "")
    if key != "ebuilder":
        return jsonify({"status": False, "code": 401, "error": "无效的key"})
    base64_image = data.get("image", "")
    confidence = data.get("confidence", 0.4)

    if base64_image:
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data))
    else:
        image = data.get("url", "")
    # 调用YOLO模型进行对象检测
    inner_model = model if version <= 2 else modelV2
    has_origin_shape = False if version <= 1 else True
    results = inner_model.predict(source=image, conf=confidence, save=True)
    # 检查是否有检测结果
    if not results:
        return jsonify(
            {"status": True, "code": 200, "data": {} if has_origin_shape else []}
        )  # 如果没有检测到对象，返回一个空对象

    # 返回检测结果
    return jsonify(
        {"status": True, "code": 200, "data": json.loads(results[0].tojson(has_origin_shape=has_origin_shape))}
    )


@app.route("/ocr", methods=["POST"])
def ocr():
    try:
        # 获取上传的图像文件
        data = json.loads(request.data)
        key = data.get("key", "")
        if key != "ebuilder":
            return jsonify({"status": False, "code": 401, "error": "无效的key"})
        base64_image = data.get("image", "")

        # 获取当前时间戳（毫秒数）
        timestamp = str(int(time.time() * 1000))
        # 构建签名
        signature = hashlib.md5((OCR_APP_ID + timestamp + OCR_APP_SECRET).encode()).hexdigest()
        # 设置请求头信息
        headers = {"appId": OCR_APP_ID, "timestamp": timestamp, "sign": signature}
        # 构建请求参数
        if base64_image:
            image_data = base64.b64decode(base64_image)
        else:
            response = requests.get(data.get("url", ""))
            if response.status_code == 200:
                image_data = response.content
            else:
                return jsonify({"status": False, "code": 500, "error": "图片下载失败"})
        files = {"img": image_data}

        # 调用OCR接口
        results = requests.post(OCR_API_URL, files=files, headers=headers)

        # 检查是否有检测结果
        if results.status_code != 200:
            return jsonify({"status": False, "code": 500, "error": str(results.content)})
        # 返回检测结果
        json_result = json.loads(results.content)
        if json_result["status_code"] == 5200:
            json_result["status_code"] = 200
        return jsonify(json_result)

    except Exception as e:
        return jsonify({"status": False, "code": 500, "error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=18000)
