import json
import urllib.request

url='http://localhost:8088/api/v1/chat'
data={"user_id":"DC537411-D831-5393-ABEE-6154CF0A6C0A","message":"عايز مرشد مناسب","language":"ar","history": []}
req=urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type':'application/json'})
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        print('STATUS', resp.status)
        print(resp.read().decode('utf-8'))
except urllib.error.HTTPError as e:
    print('HTTPERROR', e.code)
    try:
        print(e.read().decode('utf-8'))
    except Exception:
        pass
except Exception as ex:
    print('ERROR', ex)
