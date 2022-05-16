import base64
from uuid import uuid4
from synthesis import Run

curlist = {
    "$": "ដុល្លារ",
    "៛": "រៀល",
    "€": "អឺរ៉ូ",
    "¥": "យេន",
    "￥": "យន់",
    "₹": "រូពី",
    "£": "ផោន",
    "฿": "បាត",
    "₫": "ដុង",
    "₭": "គីប",
}

thisdict = {}


class TextToSpeechService:
    def __init__(self):
        pass

    @staticmethod
    async def detect(data: dict) -> dict:
        text = data["text"]
        filename = f"{uuid4()}.wav"
        path = "/app/public/" + filename
        device = data["device"] or "cpu"
        sound = data["sound"] or "male"

        await Run(text, path, device)

        data = open(path, "r").read()
        encoded = base64.b64encode(data)

        return {"base64": encoded}

