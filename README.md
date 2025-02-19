# مدل پردازش زبان طبیعی با Ollama

این پروژه از مدل **Ollama** برای پردازش و تحلیل داده‌های استخراج‌شده از فایل‌های PDF استفاده می‌کند. داده‌های استخراج‌شده در حافظه ذخیره شده و برای بهبود درک و پاسخ‌دهی به درخواست‌ها (Prompt) مورد استفاده قرار می‌گیرد.

## 🚀 ویژگی‌ها
- بارگذاری چندین فایل PDF و ایجاد یک پایگاه داده کوچک در حافظه.
- ساخت درخواست‌های پویا (Prompt) بر اساس داده‌های ورودی.
- استفاده از مدل **Qwen 2.5** برای تولید پاسخ‌ها.

## 🛠 نصب و راه‌اندازی

### 1️⃣ نصب Ollama
ابتدا **[Ollama](https://ollama.com/)** را روی سیستم خود نصب کنید.

### 2️⃣ نصب مدل Qwen 2.5
پس از نصب Ollama، مدل **Qwen 2.5** را دانلود و تنظیم کنید:
```sh
ollama run qwen2.5
```

### 3️⃣ ایجاد پوشه پروژه
```sh
mkdir my_nlp_project
cd my_nlp_project
```

### 4️⃣ ایجاد و فعال‌سازی محیط مجازی
```sh
python -m venv venv
```
**فعال‌سازی در ویندوز:**
```sh
venv\Scripts\activate
```
**فعال‌سازی در macOS/Linux:**
```sh
source venv/bin/activate
```

### 5️⃣ نصب پکیج‌های مورد نیاز
```sh
pip install -r requirements.txt
```

## ▶ اجرای پروژه
برای اجرای برنامه، دستور زیر را اجرا کنید:
```sh
python main.py
```

## 📜 وابستگی‌ها
این پروژه از پکیج‌های زیر استفاده می‌کند:
- `langchain`
- `langchain_community`
- `langchain_core`

این وابستگی‌ها به‌طور خودکار از `requirements.txt` نصب خواهند شد.

---
**📌 نکات:**
- قبل از اجرای `main.py`، اطمینان حاصل کنید که Ollama در حال اجرا است.
- در صورت بروز مشکلات مربوط به وابستگی‌ها، پیشنهاد می‌شود محیط مجازی را دوباره ایجاد کنید.

برنامه‌نویسی لذت‌بخشی داشته باشید! 🚀

