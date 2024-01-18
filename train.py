from ultralytics import YOLO
from multiprocessing import freeze_support

# Load a model
model = YOLO("F:\gsfc\project\helmet dectection and responce\Helmet Detection\\runs\detect\\train5\weights\\best.pt")  # build a new model from scratch

if __name__ == '__main__':
    # Call freeze_support() to prevent the RuntimeError on Windows
    freeze_support()

    # Your other code here
    model.train(data="F:\gsfc\project\helmet dectection and responce\Helmet Detection\images\data1.yaml", epochs=50,batch=5, device=0,imgsz=416)
