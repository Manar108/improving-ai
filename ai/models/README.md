ما يجب أن يحتويه مجلد `ai/models`

مقدمة
- هذا المجلد مخصص لنسخ الموديلات (models) وملفات التحويل (encoders/scalers) وملفات التعريف (metadata/config) اللازمة للقيام بالاستدلال داخل التطبيق.

محتويات مقترحة
- `model.joblib` أو `model.pkl` أو `model.safetensors` : ملف الوزن أو نموذج الـ ML/NN المدرب.
- `scaler.joblib` أو `scaler.pkl` : ملف لتحجيم الميزات (StandardScaler, MinMax, إلخ).
- `feature_encoder.joblib` : أي encoders (OneHotEncoder, LabelEncoder) المستخدمة.
- `tokenizer.json`, `tokenizer_config.json` : إن كان هناك NLP/LLM يعتمد توكنيزر.
- `config.json` أو `model_config.yaml` : يصف نوع النموذج، نسخة المكتبة، متطلبات التحميل، أسماء الميزات المتوقعة.
- `metadata.json` : تاريخ التدريب، الدقة، عدد الميزات، وصف الأعمدة، تحميل نسخة dataset.
- `serve.py` أو `loader.py` (اختياري) : وظيفة صغيرة توضح كيف تُحمّل النموذج داخل الكود (مثال `load_model()`).
- `requirements.txt` (اختياري) : مكتبات إضافية خاصة بالموديل.
- `README.md` (هذا الملف) : يشرح محتوى المجلد وكيفيّة الاستخدام.

مثال بسيط لملف `loader.py`:

```python
import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).parent

def load_model():
    model = joblib.load(MODEL_DIR / 'model.joblib')
    scaler = joblib.load(MODEL_DIR / 'scaler.joblib')
    return model, scaler
```

نصائح
- لا تخزني ملفات ضخمة (weights) في git؛ استخدمي تخزين خارجي أو git-lfs.
- ضعي نسخة صغيرة (dummy model) للاختبارات داخل repo إن احتجت لتشغيل اختبارات دون الحاجة لملفات ثقيلة.
- احتفظي بإصدار الموديل (`v1`, `v2`) في اسم الملف أو في `metadata.json`.

إذا تحبي، أقدر:
- أضيف ملف `loader.py` فعليًا بمثال تحميل جاهز،
- أو أنقل بعض artifacts الموجودة في `data/artifacts/` إلى هنا وأعدّل مسارات التحميل في الكود.