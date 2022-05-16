import os
from fastapi import FastAPI

from apis import main_router


app = FastAPI()

# @app.exception_handler(AppExceptionCase)
# async def custom_app_exception_handler(request, e):
#     return await app_exception_handler(request, e)


app.include_router(main_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=os.environ.get("PORT") or 8000,
        reload=os.environ.get("APP_ENV") != "production",
    )

