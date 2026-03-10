import os
from config import Config
from modules.utils import send_alert_email
import numpy as np

print('ENV SENDER_EMAIL =', os.getenv('SENDER_EMAIL'))
print('ENV SENDER_PASSWORD =', '***' if os.getenv('SENDER_PASSWORD') else '(none)')
print('Config SENDER_EMAIL =', Config.SENDER_EMAIL)
print('Config SENDER_PASSWORD =', '***' if Config.SENDER_PASSWORD else '(none)')

frame = np.zeros((100, 100, 3), dtype=np.uint8)
try:
    send_alert_email(Config.SENDER_EMAIL, frame, person_id=1, person_count=1)
    print('send_alert_email succeeded (no exception)')
except Exception as e:
    print('send_alert_email exception:', type(e), e)
