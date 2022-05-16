from fastapi import APIRouter

from apis.text_to_speech.services.text_to_speech import TextToSpeechService

router = APIRouter(prefix="/text-to-speech")


@router.post("/", response_model=dict)
async def findAll(body: dict):

    return await TextToSpeechService.detect(body)

