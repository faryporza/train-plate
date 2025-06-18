import os
from ultralytics import YOLO
import yaml
import argparse
import glob
import torch

def check_previous_training(project_path, name):
    """ตรวจสอบการเทรนก่อนหน้า และแสดงข้อมูล"""
    run_dir = os.path.join(project_path, name)
    weights_dir = os.path.join(run_dir, "weights")
    
    if not os.path.exists(run_dir):
        return False, 0, None
    
    # ตรวจสอบโมเดลที่บันทึกไว้
    best_model = os.path.join(weights_dir, "best.pt")
    last_model = os.path.join(weights_dir, "last.pt")
    
    model_path = None
    if os.path.exists(last_model):
        model_path = last_model
    elif os.path.exists(best_model):
        model_path = best_model
    
    # พยายามดึงข้อมูล epoch จาก results.csv
    results_csv = os.path.join(run_dir, "results.csv")
    last_epoch = 0
    
    if os.path.exists(results_csv):
        import csv
        try:
            with open(results_csv, 'r') as f:
                csv_reader = csv.reader(f)
                rows = list(csv_reader)
                if len(rows) > 1:  # มีเฮดเดอร์ + ข้อมูลอย่างน้อย 1 แถว
                    last_epoch = int(rows[-1][0])  # ดึง epoch จากคอลัมน์แรกของแถวสุดท้าย
        except:
            pass
    
    return os.path.exists(run_dir), last_epoch, model_path

def main():
    # เพิ่มตัวแปรรับค่าพารามิเตอร์จาก command line สำหรับการเทรนต่อ
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='เทรนต่อจากโมเดลล่าสุด')
    parser.add_argument('--epochs', type=int, default=100, help='จำนวน epochs ทั้งหมดที่ต้องการเทรน')
    parser.add_argument('--device', type=str, default='', help='กำหนดอุปกรณ์ที่ใช้ในการเทรน (เช่น cuda:0, cpu)')
    args = parser.parse_args()
    
    print("Starting YOLOv11 License Plate Detection Training...")
    
    # ตรวจสอบ GPU
    if torch.cuda.is_available():
        device = "cuda:0" if not args.device else args.device
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n✅ พบ GPU: {gpu_name} ({gpu_memory:.2f} GB)")
        print(f"✅ จำนวน GPUs: {torch.cuda.device_count()}")
    else:
        device = "cpu"
        print("\n⚠️ ไม่พบ GPU - จะใช้ CPU ในการเทรน (จะช้ากว่ามาก)")
    
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
    
    # โฟลเดอร์สำหรับโปรเจค
    project_dir = "runs/detect"
    run_name = "license_plate_train"
    
    # ตรวจสอบการเทรนก่อนหน้า
    has_previous, last_epoch, model_path = check_previous_training(project_dir, run_name)
    
    # แสดงข้อความเกี่ยวกับการเทรนก่อนหน้า
    if has_previous:
        print("\n" + "="*50)
        print(f"📂 พบการเทรนก่อนหน้าที่ {os.path.join(project_dir, run_name)}")
        print(f"⏱️ เทรนไปแล้ว {last_epoch} epochs")
        
        if model_path:
            print(f"💾 พบไฟล์โมเดลที่ {model_path}")
        else:
            print("⚠️ ไม่พบไฟล์โมเดล best.pt หรือ last.pt")
        
        if args.resume:
            print(f"✅ กำลังจะเทรนต่อจาก epoch {last_epoch+1} ถึง {args.epochs}")
        else:
            print("❌ ไม่ได้เลือกโหมดเทรนต่อ (--resume) การเทรนจะเริ่มใหม่จาก epoch 1")
            print("   หากต้องการเทรนต่อ ให้รันด้วยคำสั่ง: python train.py --resume --epochs 150")
        print("="*50 + "\n")
    
    # Initialize YOLOv11 model
    if args.resume and model_path:
        print(f"โหลดโมเดลจาก {model_path} เพื่อเทรนต่อ...")
        model = YOLO(model_path)
    else:
        print("เริ่มเทรนใหม่ด้วยโมเดล yolo11n.pt...")
        model = YOLO('yolo11n.pt')  # Start with nano model for faster training
    
    # กำหนดขนาด batch ตามความจำของ GPU
    batch_size = 16
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_mem < 6:  # น้อยกว่า 6GB
            batch_size = 8
            print(f"⚠️ GPU มีหน่วยความจำน้อย ({gpu_mem:.1f}GB) ปรับ batch size เป็น {batch_size}")
        elif gpu_mem >= 12:  # 12GB หรือมากกว่า
            batch_size = 32
            print(f"✅ GPU มีหน่วยความจำสูง ({gpu_mem:.1f}GB) ปรับ batch size เป็น {batch_size}")
    
    # Training parameters
    training_params = {
        # พารามิเตอร์พื้นฐานสำหรับการเทรน
        'data': data_yaml,           # ไฟล์ config ข้อมูลสำหรับการเทรนโมเดล
        'epochs': args.epochs,       # จำนวนรอบในการเทรนทั้งหมด (รวมกับที่เทรนก่อนหน้า ถ้า resume=True)
        'imgsz': 640,                # ขนาดของรูปภาพที่ใช้ในการเทรน (พิกเซล)
        'batch': batch_size,         # จำนวนรูปภาพต่อการประมวลผลหนึ่งครั้ง (ปรับตามขนาด GPU)
        
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
        'device': device,            # อุปกรณ์ที่ใช้ในการเทรน (cuda:0, cpu)
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
        'resume': args.resume,       # ทำการเทรนต่อจากครั้งก่อน (รับค่าจาก command line)
        'amp': True,                 # ใช้ mixed precision training (เร่งความเร็ว GPU)
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
        'half': torch.cuda.is_available(),  # ใช้ FP16 half-precision inference เมื่อมี GPU
        'dnn': False,                # ใช้ OpenCV DNN สำหรับ ONNX inference
        'plots': True,               # สร้างกราฟและรูปภาพผลลัพธ์
    }
    
    print("Starting training with the following parameters:")
    print(f"  device: {device} {'(GPU)' if device.startswith('cuda') else '(CPU)'}")
    print(f"  batch size: {batch_size}")
    for key, value in training_params.items():
        print(f"  {key}: {value}")
    
    # Start training
    try:
        # ตรวจสอบการใช้หน่วยความจำของ GPU ก่อนเทรน
        if device.startswith('cuda'):
            # แสดงข้อมูล GPU ก่อนเริ่มเทรน
            free_mem = torch.cuda.memory_reserved(0)/1024**2
            print(f"GPU memory reserved: {free_mem:.1f} MB")
            print("Clearing GPU cache to free up memory...")
            torch.cuda.empty_cache()
            
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
        
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("\n❌ เกิดข้อผิดพลาด: GPU มีหน่วยความจำไม่เพียงพอ")
            print("💡 ลองลดค่า batch size ลง เช่น --batch 8 หรือ --batch 4")
            print("💡 หรือลดขนาดรูปภาพ เช่น --imgsz 416")
        else:
            print(f"Error during training: {str(e)}")
        return
    
    # แสดงคำอธิบายเพิ่มเติมเกี่ยวกับการเทรนต่อ
    print("\n" + "="*80)
    print("📝 คำแนะนำในการเทรนต่อ (Resume Training)")
    print("="*80)
    print("✅ หากต้องการเทรนต่อในอนาคต สามารถใช้คำสั่งนี้:")
    print(f"   python train.py --resume --epochs {args.epochs + 50}")
    print()
    print("🔎 คำอธิบายการเทรนต่อ:")
    print("• โมเดลจะโหลดไฟล์ last.pt หรือ best.pt จากโฟลเดอร์ runs/detect/license_plate_train/weights/")
    print(f"• จะเทรนต่อจาก epoch {args.epochs + 1} ไปจนถึงจำนวน epochs ที่กำหนดใหม่")
    print("• ประวัติการเทรนทั้งหมดจะถูกบันทึกต่อในไฟล์ results.csv")
    print()
    print("⚠️ ข้อควรระวัง:")
    print("• ห้ามลบหรือย้ายโฟลเดอร์ runs/detect/license_plate_train/ มิฉะนั้นจะไม่สามารถเทรนต่อได้")
    print("• ถ้ามีการเปลี่ยนแปลง data.yaml หรือชุดข้อมูล อาจทำให้การเทรนต่อมีปัญหาได้")
    print("="*80)
    
    print("\nTraining pipeline completed!")

if __name__ == "__main__":
    main()
