import asyncio
import os.path
import random
import time
import csv

import uvicorn
import aiofiles

from fastapi import FastAPI, Request, UploadFile, File

from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from typing import List, Dict
from models import AnalyzeResult

from uuid import uuid4

from Test_track import get_result

app = FastAPI()
TMP_UPLOADS_DIRECTORY = 'uploads'

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "id": id})


@app.post("/download")
async def download(data: List[List[str]]) -> FileResponse:
    # Путь к файлу, который нужно отправить для скачивания
    filename = str(uuid4()) + '.csv'
    file_path = os.path.join(TMP_UPLOADS_DIRECTORY, filename)
    with open(file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data)
    print(file_path)

    # Отправить файл для скачивания
    return FileResponse(file_path, filename=filename)


@app.post("/upload")
async def upload_video(videos: List[UploadFile] = File(...),
                       jsonFiles: List[UploadFile] = File(...)) -> List[List]:
    if TMP_UPLOADS_DIRECTORY not in os.listdir():
        os.mkdir(TMP_UPLOADS_DIRECTORY)

    videos_list = []
    jsonFiles_list = []

    for video in videos:
        filename = str(uuid4()) + '.mp4'
        videos_list.append(os.path.join(TMP_UPLOADS_DIRECTORY, filename))
        async with aiofiles.open(os.path.join(TMP_UPLOADS_DIRECTORY, filename), 'wb') as out_file:
            content = await video.read()  # async read
            await out_file.write(content)  # async write

    for file in jsonFiles:
        filename = str(uuid4()) + '.json'
        jsonFiles_list.append(os.path.join(TMP_UPLOADS_DIRECTORY, filename))
        async with aiofiles.open(os.path.join(TMP_UPLOADS_DIRECTORY, filename), 'wb') as out_file:
            content = await file.read()  # async read
            await out_file.write(content)  # async write

    result = get_result(videos_list, jsonFiles_list)

    # Возвращаем результаты проверки в формате JSON
    # results = [{
    #     "file_name": "KRA-2-7-2023-08-22-evening",
    #     "car": "car",
    #     "quantity_car": random.randint(100, 1000),
    #     "average_speed_car": 32.64,
    #     "van": "van",
    #     "quantity_van": 15,
    #     "average_speed_van": 25.30,
    #     "bus": "bus",
    #     "quantity_bus": 22,
    #     "average_speed_bus": 26.75
    # }]
    return result


if __name__ == '__main__':
    uvicorn.run(app, port=3000)
