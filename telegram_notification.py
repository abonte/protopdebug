import dotenv
import os

import requests


dotenv.load_dotenv()


def send_telegram_notification(text):
    payload = {
        'chat_id': os.getenv('TELEGRAM_CHAT_ID'),
        'text': text,
        'parse_mode': 'HTML'
    }
    return requests.post(
        f'https://api.telegram.org/bot{os.getenv("TELEGRAM_TOKEN")}/sendMessage',
        data=payload).content
