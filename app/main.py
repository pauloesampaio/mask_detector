from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import orjson
from utils.io_utils import yaml_loader, load_string_to_image
from utils.model_utils import predict
from tensorflow.keras.models import load_model
import datetime

try:
    config = yaml_loader("./config/config.yml")
    model = load_model(config["paths"]["model_path"])
except IOError as e:
    errno, strerror = e.args
    print("Error loading config or model({0}): {1}".format(errno, strerror))


class InputImage(BaseModel):
    image_string: str


app = FastAPI()


@app.get("/")
def check_api():
    return {"API status": "Up and running!"}


@app.post("/predict/")
async def get_prediction(input_image: InputImage):
    try:
        image = load_string_to_image(input_image.image_string)
        predictions = predict(model, image, config)
        predictions = orjson.loads(
            orjson.dumps(predictions, option=orjson.OPT_SERIALIZE_NUMPY)
        )
    except IOError:
        predictions = None
        raise HTTPException(status_code=300, detail="Download error")
    response = {
        "predictions": predictions,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(response)
    return response
