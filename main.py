import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from query.query_api import router as query_router

app = FastAPI(
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/documentation",
    redoc_url="/api/v1/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type"],
    max_age=86400,
)


@app.exception_handler(Exception)
async def http_exception_handler(request, exc):
    logging.error(
        f"Unexpected error occured: {exc.reason}",
        exc_info=True,
    )
    message = str(exc.detail)
    return JSONResponse({"message": message}, status_code=exc.status_code)


# app.include_router(mapping_api_router)
# app.include_router(preview_router)
# app.include_router(dwh_api_router)
# # app.include_router(visualisation_router)
app.include_router(query_router)
# app.include_router(thread_router)

@app.get("/")
def hello_world():
    return "Hello, World!"
