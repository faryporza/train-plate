# การฝึกโมเดล YOLOv11 สำหรับตรวจจับป้ายทะเบียนรถ

โปรเจกต์นี้ใช้สำหรับฝึกโมเดล YOLOv11 เพื่อตรวจจับป้ายทะเบียนรถโดยใช้ชุดข้อมูลจาก Roboflow

## ข้อกำหนดเบื้องต้น

- Python 3.8 หรือสูงกว่า
- GPU ที่รองรับ CUDA (แนะนำสำหรับการฝึกที่เร็วขึ้น)
- Git (สำหรับ clone repositories)

## การติดตั้ง

### 1. สร้าง Virtual Environment (แนะนำ)

```bash
# สร้าง virtual environment
python -m venv yolo_env

# เปิดใช้งาน virtual environment
# บน Windows:
yolo_env\Scripts\activate
# บน macOS/Linux:
source yolo_env/bin/activate
```

### 2. ติดตั้งไลบรารีที่จำเป็น

```bash
# ติดตั้ง PyTorch พร้อมการรองรับ CUDA (สำหรับการฝึกด้วย GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# ติดตั้ง Ultralytics YOLOv11
pip install ultralytics

# ติดตั้ง dependencies เพิ่มเติม
pip install opencv-python
pip install pillow
pip install matplotlib
pip install seaborn
pip install pandas
pip install pyyaml
pip install tqdm
pip install psutil
pip install thop
```

### 3. ติดตั้ง Roboflow (หากต้องการดาวน์โหลดชุดข้อมูล)

```bash
pip install roboflow
```

### 4. ตรวจสอบการติดตั้ง

```bash
# ตรวจสอบว่า YOLO ถูกติดตั้งอย่างถูกต้องหรือไม่
yolo version

# ตรวจสอบ PyTorch และการใช้งาน CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA พร้อมใช้งาน: {torch.cuda.is_available()}')"
```

## การติดตั้งอย่างรวดเร็ว (ทั้งหมดในครั้งเดียว)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python pillow matplotlib seaborn pandas pyyaml tqdm psutil thop roboflow
```

## การติดตั้งสำหรับ CPU เท่านั้น (ไม่มี GPU)

หากคุณไม่มี GPU ที่รองรับ CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics opencv-python pillow matplotlib seaborn pandas pyyaml tqdm psutil thop roboflow
```

## โครงสร้างโปรเจกต์

```
d:\train\
├── train.py                    # สคริปต์หลักสำหรับการฝึก
├── README_TH.md               # ไฟล์นี้
├── My-First-Project-1/         # ชุดข้อมูลที่ดาวน์โหลดจาก Roboflow
│   ├── data.yaml              # การกำหนดค่าชุดข้อมูล
│   ├── train/                 # รูปภาพและ labels สำหรับการฝึก
│   │   ├── images/
│   │   └── labels/
│   ├── valid/                 # รูปภาพและ labels สำหรับการตรวจสอบ
│   │   ├── images/
│   │   └── labels/
│   └── test/                  # รูปภาพและ labels สำหรับการทดสอบ (หากมี)
│       ├── images/
│       └── labels/
└── runs/                      # ผลลัพธ์การฝึก (สร้างหลังจากการฝึก)
    └── detect/
        └── license_plate_train/
```

## การใช้งาน

### 1. ดาวน์โหลดชุดข้อมูล (หากยังไม่ได้ทำ)

ชุดข้อมูลควรถูกดาวน์โหลดไว้ในโฟลเดอร์ `My-First-Project-1` แล้ว

### 2. เริ่มการฝึก

```bash
cd d:\train
python train.py
```

### 3. ติดตามการฝึก

- ความคืบหน้าการฝึกจะแสดงในคอนโซล
- Tensorboard logs และกราฟการฝึกจะถูกบันทึกใน `runs/detect/license_plate_train/`
- น้ำหนักโมเดลที่ดีที่สุดจะถูกบันทึกเป็น `best.pt`

### 4. ดูผลลัพธ์

หลังจากการฝึก คุณจะพบ:
- **โมเดลที่ดีที่สุด**: `runs/detect/license_plate_train/weights/best.pt`
- **โมเดลล่าสุด**: `runs/detect/license_plate_train/weights/last.pt`
- **กราฟการฝึก**: `runs/detect/license_plate_train/`
- **ผลการตรวจสอบ**: แสดงในคอนโซลและเมตริกที่บันทึกไว้

## การกำหนดค่าการฝึก

สคริปต์การฝึกใช้พารามิเตอร์หลักเหล่านี้:
- **Epochs**: 100 (ปรับตามความต้องการ)
- **Batch size**: 16 (ลดหากมีปัญหาเรื่องหน่วยความจำ)
- **ขนาดรูปภาพ**: 640x640
- **โมเดล**: YOLOv11 Nano (เร็วที่สุด, สามารถเปลี่ยนเป็น 's', 'm', 'l', 'x' เพื่อความแม่นยำที่ดีขึ้น)

## การแก้ไขปัญหา

### ปัญหาที่พบบ่อย

1. **CUDA Out of Memory**
   ```bash
   # ลด batch size ในไฟล์ train.py
   'batch': 8  # หรือเล็กกว่านี้
   ```

2. **ข้อผิดพลาด Module not found**
   ```bash
   # ตรวจสอบว่า virtual environment เปิดใช้งานแล้ว
   # ติดตั้ง packages ใหม่
   pip install --upgrade ultralytics
   ```

3. **ข้อผิดพลาดสิทธิ์บน Windows**
   ```bash
   # เรียกใช้ command prompt ในฐานะผู้ดูแลระบบ
   ```

### ความต้องการของระบบ

- **RAM ขั้นต่ำ**: 8GB
- **RAM ที่แนะนำ**: 16GB+
- **GPU VRAM**: 4GB+ (สำหรับ batch size 16)
- **พื้นที่จัดเก็บ**: 5GB+ ว่าง

## การใช้โมเดลสำหรับพยากรณ์

หลังจากการฝึก ใช้โมเดลของคุณสำหรับการพยากรณ์:

```python
from ultralytics import YOLO

# โหลดโมเดลที่ฝึกแล้ว
model = YOLO('runs/detect/license_plate_train/weights/best.pt')

# รันการพยากรณ์
results = model('path/to/your/image.jpg')
results[0].show()  # แสดงผลลัพธ์
```

## แหล่งข้อมูลเพิ่มเติม

- [เอกสาร Ultralytics](https://docs.ultralytics.com/)
- [YOLOv11 GitHub](https://github.com/ultralytics/ultralytics)
- [เอกสาร Roboflow](https://docs.roboflow.com/)

## การสนับสนุน

หากพบปัญหา:
1. ตรวจสอบข้อความแสดงข้อผิดพลาดในคอนโซล
2. ตรวจสอบว่า dependencies ทั้งหมดถูกติดตั้งอย่างถูกต้อง
3. ตรวจสอบว่ารูปแบบชุดข้อมูลตรงกับข้อกำหนดของ YOLOv11
4. ตรวจสอบหน่วยความจำ GPU ที่มีอยู่และลด batch size หากจำเป็น
