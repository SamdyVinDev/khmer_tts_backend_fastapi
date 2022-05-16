from fastapi import APIRouter
from apis.text_to_speech.controller import router as text_to_speech_router

main_router = APIRouter()
main_router.include_router(text_to_speech_router)
