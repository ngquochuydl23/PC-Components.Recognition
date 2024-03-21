from fastapi import FastAPI
from typing import Union

app = FastAPI()


@app.get("/ping")
def pong():
    return {"ping": "pong!"}
