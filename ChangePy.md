เยี่ยมเลยค่ะ! ตอนนี้เครื่องคุณ **มี Python 3.10 แล้วแน่นอน** ✅
อยู่ที่:

```
/opt/homebrew/opt/python@3.10/bin/python3.10
```

---

### ✅ ขั้นตอนต่อไป (แนะนำให้เริ่มใหม่อย่างสะอาด):

#### 1. ออกจาก venv เดิมก่อน:

```bash
deactivate
```

#### 2. สร้าง venv ใหม่ด้วย Python 3.10:

```bash
/opt/homebrew/opt/python@3.10/bin/python3.10 -m venv venv310
```

#### 3. เข้าใช้งาน venv ที่สร้างใหม่:

```bash
source venv310/bin/activate
```

#### 4. ตรวจสอบว่าใช้ Python 3.10 แล้ว:

```bash
python --version
# ✅ ต้องได้ Python 3.10.x
```

---

### ✅ ติดตั้งแพ็กเกจสำหรับโปรเจกต์ CRNN:

```bash
pip install --upgrade pip
pip install tensorflow==2.3.0 keras==2.4.3 opencv-python-headless numpy
```

> ⚠️ ใช้ `opencv-python-headless` แทน `opencv-python` เพราะบน macOS M1/M2 อาจเจอปัญหา GUI library

---

### 🎁 ถ้ามี `requirements.txt` อยู่แล้ว:

แก้ไฟล์ให้รองรับ Python 3.10 แล้วติดตั้งด้วย:

```bash
pip install -r requirements.txt
```

หากอยากให้ช่วยปรับ `requirements.txt` ให้เข้ากับ Python 3.10 และ TensorFlow 2.3 บอกมาได้เลยนะคะ จะจัดให้เลยค่ะ 💪
