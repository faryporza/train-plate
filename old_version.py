import os
from ultralytics import YOLO
import yaml

def main():
    print("Starting YOLOv11 License Plate Detection Training...")
    
    # Set up paths
    dataset_path = "My-First-Project-1"
    data_yaml = os.path.join(dataset_path, "data.yaml")
    
    # Verify dataset exists
    if not os.path.exists(data_yaml):
        print(f"Error: Dataset configuration file not found at {data_yaml}")
        return
    
    # Load and verify data.yaml
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"Dataset classes: {data_config['names']}")
    print(f"Number of classes: {data_config['nc']}")
    
    # Initialize YOLOv11 model
    # You can use different model sizes: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    model = YOLO('yolo11n.pt')  # Start with nano model for faster training
    
    # Training parameters
    training_params = {
        # พารามิเตอร์พื้นฐานสำหรับการเทรน
        'data': data_yaml,           # ไฟล์ config ข้อมูลสำหรับการเทรนโมเดล
        'epochs': 100,               # จำนวนรอบในการเทรน
        'imgsz': 640,                # ขนาดของรูปภาพที่ใช้ในการเทรน (พิกเซล)
        'batch': 16,                 # จำนวนรูปภาพต่อการประมวลผลหนึ่งครั้ง
        
        # พารามิเตอร์สำหรับการปรับค่า Learning Rate
        'lr0': 0.01,                 # ค่า learning rate เริ่มต้น
        'lrf': 0.1,                  # ค่า learning rate สุดท้าย (คูณกับ lr0)
        'momentum': 0.937,           # ค่า momentum สำหรับ optimizer
        'weight_decay': 0.0005,      # ค่า weight decay สำหรับป้องกันการ overfitting
        'warmup_epochs': 3,          # จำนวนรอบการเทรนที่ใช้สำหรับ warmup
        'warmup_momentum': 0.8,      # ค่า momentum เริ่มต้นในช่วง warmup
        'warmup_bias_lr': 0.1,       # ค่า bias learning rate ในช่วง warmup
        
        # พารามิเตอร์สำหรับการคำนวณ Loss
        'box': 7.5,                  # น้ำหนักของ box loss
        'cls': 0.5,                  # น้ำหนักของ class loss
        'dfl': 1.5,                  # น้ำหนักของ DFL loss
        'pose': 12.0,                # น้ำหนักของ pose loss (สำหรับการตรวจจับท่าทาง)
        'kobj': 1.0,                 # น้ำหนักของ keypoint-object loss
        'label_smoothing': 0.0,      # ค่า label smoothing (ช่วยลดการ overfitting)
        'nbs': 64,                   # ค่า nominal batch size สำหรับการปรับพารามิเตอร์
        
        # พารามิเตอร์สำหรับการปรับแต่งรูปภาพ (Data Augmentation)
        'hsv_h': 0.015,              # การปรับค่าสี Hue
        'hsv_s': 0.7,                # การปรับค่าความอิ่มตัวของสี (Saturation)
        'hsv_v': 0.4,                # การปรับค่าความสว่าง (Value)
        'degrees': 0.0,              # การหมุนภาพ (องศา)
        'translate': 0.1,            # การเลื่อนตำแหน่งภาพ
        'scale': 0.5,                # การปรับขนาดภาพ
        'shear': 0.0,                # การเฉือนภาพ
        'perspective': 0.0,          # การปรับมุมมองภาพ
        'flipud': 0.0,               # โอกาสในการพลิกภาพจากบนลงล่าง
        'fliplr': 0.5,               # โอกาสในการพลิกภาพจากซ้ายไปขวา
        'mosaic': 1.0,               # โอกาสในการใช้เทคนิค mosaic
        'mixup': 0.0,                # โอกาสในการใช้เทคนิค mixup
        'copy_paste': 0.0,           # โอกาสในการใช้เทคนิค copy-paste
        'auto_augment': 'randaugment', # ใช้การเพิ่มข้อมูลแบบอัตโนมัติด้วย RandAugment
        'erasing': 0.4,              # โอกาสในการลบบางส่วนของภาพ
        'crop_fraction': 1.0,        # สัดส่วนของการครอปภาพ
        
        # พารามิเตอร์สำหรับการบันทึกและการแคช
        'save': True,                # บันทึกโมเดล
        'save_period': -1,           # บันทึกโมเดลทุกๆ n epochs (-1 = บันทึกเฉพาะโมเดลที่ดีที่สุด)
        'cache': False,              # แคชข้อมูลในหน่วยความจำ
        
        # พารามิเตอร์สำหรับฮาร์ดแวร์และการประมวลผล
        'device': '',                # อุปกรณ์ที่ใช้ในการเทรน (ว่างเปล่า = เลือกอัตโนมัติ)
        'workers': 8,                # จำนวน worker สำหรับการโหลดข้อมูล
        'project': 'runs/detect',    # โฟลเดอร์โปรเจค
        'name': 'license_plate_train', # ชื่อการเทรนนี้
        'exist_ok': False,           # อนุญาตให้เขียนทับโฟลเดอร์เดิม
        
        # พารามิเตอร์สำหรับโมเดล
        'pretrained': True,          # ใช้โมเดลที่ผ่านการเทรนมาแล้ว
        'optimizer': 'auto',         # ตัวเลือก optimizer (auto, SGD, Adam, etc.)
        'verbose': True,             # แสดงข้อมูลโดยละเอียด
        'seed': 0,                   # ค่า seed สำหรับสร้างตัวเลขสุ่ม
        'deterministic': True,       # ใช้การคำนวณแบบ deterministic
        'single_cls': False,         # เทรนโมเดลด้วยคลาสเดียว
        'rect': False,               # ใช้การเทรนแบบ rectangular
        'cos_lr': False,             # ใช้ cosine learning rate scheduler
        'close_mosaic': 10,          # ปิดการใช้ mosaic ก่อนจบการเทรน n epochs
        'resume': False,             # ทำการเทรนต่อจากครั้งก่อน
        'amp': True,                 # ใช้ mixed precision training
        'fraction': 1.0,             # สัดส่วนของชุดข้อมูลที่จะใช้
        'profile': False,            # แสดงข้อมูล profile ของโค้ด
        'freeze': None,              # แช่แข็งเลเยอร์ (ไม่อัปเดตค่า weight)
        
        # พารามิเตอร์สำหรับโมเดลขั้นสูง
        'multi_scale': False,        # ใช้การเทรนแบบ multi-scale
        'overlap_mask': True,        # อนุญาตให้มาส์กทับซ้อนกัน
        'mask_ratio': 4,             # อัตราส่วนการลดขนาดมาส์ก
        'dropout': 0.0,              # ค่า dropout rate
        
        # พารามิเตอร์สำหรับการทดสอบ
        'val': True,                 # ทดสอบบนชุดข้อมูล validation
        'split': 'val',              # ชุดข้อมูลสำหรับการทดสอบ (train, val, test)
        'save_json': False,          # บันทึกผลลัพธ์เป็นไฟล์ JSON
        'save_hybrid': False,        # บันทึกผลลัพธ์แบบ hybrid
        'conf': None,                # ค่าความเชื่อมั่นขั้นต่ำสำหรับการตรวจจับ
        'iou': 0.7,                  # ค่า IoU threshold
        'max_det': 300,              # จำนวนวัตถุสูงสุดที่ตรวจจับได้ต่อรูป
        'half': False,               # ใช้ FP16 half-precision inference
        'dnn': False,                # ใช้ OpenCV DNN สำหรับ ONNX inference
        'plots': True,               # สร้างกราฟและรูปภาพผลลัพธ์
    }
    
    print("Starting training with the following parameters:")
    for key, value in training_params.items():
        print(f"  {key}: {value}")
    
    # Start training
    try:
        results = model.train(**training_params)
        print("\nTraining completed successfully!")
        print(f"Best model saved to: {model.trainer.best}")
        
        # Evaluate the model
        print("\nEvaluating model on validation set...")
        metrics = model.val()
        
        print(f"\nValidation Results:")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        
        # Export the model to different formats
        print("\nExporting model...")
        model.export(format='onnx')  # Export to ONNX format
        print("Model exported to ONNX format")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return
    
    print("\nTraining pipeline completed!")

if __name__ == "__main__":
    main()
