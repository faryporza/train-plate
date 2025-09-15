# สิ่งที่ต้องติดตั้ง 

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
## การใช้งาน

### 1. ดาวน์โหลดชุดข้อมูล (หากยังไม่ได้ทำ)

ชุดข้อมูลควรถูกดาวน์โหลดไว้ในโฟลเดอร์ `My-First-Project-1` แล้ว

### 2. เริ่มการฝึก
# เริ่มเทรนใหม่ทั้งหมด
จากไฟล์ `trainGPU.py`
```bash
python train.py --epochs 100
```
# เทรนต่อจากโมเดลเดิม
```bash
python train.py --resume --epochs 150
```
