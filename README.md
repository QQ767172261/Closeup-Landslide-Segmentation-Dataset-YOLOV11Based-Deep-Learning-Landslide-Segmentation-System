近距离拍摄滑坡分割数据集，510张，yolo，voc，coco三种标注方式
图像尺寸:640*640![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/15496425064b4bb69e3963c68321057b.png)

类别数量:2类
训练集图像数量:459; 验证集图像数量:25； 测试集图像数量:26
类别名称: 每一类图像数 ，每一类标注数![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e69b8e3f2556408abffb89ba727dcb3b.png)

landslide debris - 滑坡碎屑：456、529
landslide scar - 滑坡痕迹（或滑坡残痕）：510、571
image num: 510![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/da92d8c0f9c04cb887b9dd21783a6e10.png)


模型代码：模型训练使用yolov11n-seg训练，80个epoch训练结果，map如描述图所示。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ec5d2a71a18848fd8bbf20e18b0bb152.png)

qt界面：运行界面采用pyqt5编写，本项目已经训练好模型，配置好环境后可直接使用，运行效果见描述图像
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/161391b89791417ca957da2209d83c90.png)


# 近距离拍摄滑坡分割数据集
| 项目 | 详情 |
| ---- | ---- |
| 数据集名称 | 近距离拍摄滑坡分割数据集 |
| 总图像数量 | 510张 |
| 图像尺寸 | 640 × 640 |
| 标注格式 | YOLO、VOC、COCO 三种标注 |
| 类别总数 | 2类 |
| 训练集 | 459张 |
| 验证集 | 25张 |
| 测试集 | 26张 |

### 类别明细![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/eb9482be164b42e0bcc8a6bec5422721.png)

| 类别ID | 类别名称 | 包含该类图片数 | 标注实例数 |
| ---- | ---- | ---- | ---- |
| 0 | landslide debris - 滑坡碎屑 | 456 | 529 |
| 1 | landslide scar - 滑坡残痕 | 510 | 571 |

## 数据集摘要
本数据集为近距离实地采集滑坡场景分割数据集，一共510张640×640图像，包含滑坡碎屑、滑坡残痕2类分割目标，同时提供YOLO、VOC、COCO三种格式标注。采用YOLOv11n-seg完成80epoch分割模型训练，配套PyQt5可视化推理界面，环境配置完成后可直接加载权重进行滑坡分割检测。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5eca8e4e78144f88ab5d4f24b5ac8c8d.png)


## 应用场景
- 地质灾害现场滑坡识别、滑坡区域分割提取
- 地质巡检、边坡隐患现场识别
- 滑坡图像分割算法研究、地质灾害AI检测系统开发

## 应用价值![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cd2ab408fe284db5a477a4e052aa1cb0.png)

1. 近距离实拍滑坡场景，纹理特征清晰，适合现场地质识别；
2. 多格式标注，适配不同框架训练、转换需求；
3. 提供YOLOv11分割训练权重+PyQt5推理界面，开箱即用；
4. 同时区分滑坡碎屑与滑坡残痕，支持精细化滑坡范围分析。

## 标签
`滑坡分割` `地质灾害` `YOLOv11-seg` `实例分割` `PyQt5检测界面`

---
## 一、YOLOv11n-seg 训练简易代码
```python
from ultralytics import YOLO

# 加载分割模型
model = YOLO("yolo11n-seg.pt")

# 开始训练
model.train(
    data="landslide_seg.yaml",
    epochs=80,
    imgsz=640,
    batch=8,
    device=0,
    project="runs/seg",
    name="landslide_v11n_seg"
)
```
`landslide_seg.yaml`
```yaml
path: ./landslide_dataset
train: images/train
val: images/val
test: images/test

names:
  0: landslide debris
  1: landslide scar
```

## 二、PyQt5 推理界面简易代码
```python
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from ultralytics import YOLO

class LandslideSegUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("滑坡分割检测系统")
        self.model = YOLO("./runs/seg/landslide_v11n_seg/weights/best.pt")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.btn_open = QPushButton("选择图片检测")
        self.btn_open.clicked.connect(self.detect_image)
        self.label_img = QLabel()
        layout.addWidget(self.btn_open)
        layout.addWidget(self.label_img)
        self.setLayout(layout)

    def detect_image(self):
        path, _ = QFileDialog.getOpenFileName()
        if not path:
            return
        res = self.model(path)[0]
        img = res.plot()
        h,w,c = img.shape
        qimg = QImage(img.data, w, h, w*c, QImage.Format_RGB888)
        self.label_img.setPixmap(QPixmap.fromImage(qimg))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = LandslideSegUI()
    win.show()
    sys.exit(app.exec_())
```


