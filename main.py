from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.videos import router as videos_router
from services.video_service import ensure_dirs


def create_app() -> FastAPI:
    ensure_dirs()

    app = FastAPI(title="Detection of groups of people in video", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost",
            "http://127.0.0.1",
            "http://localhost:5173",
            "http://localhost:3000",
        ],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(videos_router)

    return app


app = create_app()
