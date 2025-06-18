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

# เริ่มเทรนใหม่ทั้งหมด
python train.py --epochs 100

# เทรนต่อจากโมเดลเดิม
python train.py --resume --epochs 150
