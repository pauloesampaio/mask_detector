import json
import yaml
import os
import requests
import base64
from io import BytesIO
from PIL import Image
import cv2


def txt_loader(path, skip_lines=0):
    """Loads txt from a path

    Args:
        path (str): path to txt file
        skip_lines (int, optional): Number of lines to skip. Defaults to 0.

    Returns:
        [list]: List with the content of the txt file, one line per row
    """
    with open(path, "r") as f:
        content = f.read().splitlines()
    return content[skip_lines:]


def json_loader(json_path):
    """Loads json from a path

    Args:
        json_path (str): Path to json file

    Returns:
        dict: Dict with json content
    """
    with open(json_path, "r") as f:
        loaded_json = json.load(f)
    return loaded_json


def yaml_loader(yaml_path):
    """Loads yaml from a path

    Args:
        yaml_path (str): Path to yaml file

    Returns:
        dict: Dict with yaml content
    """
    with open(yaml_path, "r") as f:
        loaded_yaml = yaml.safe_load(f)
    return loaded_yaml


def check_if_exists(path, create=True):
    """Checks if a path exists and, if wanted, creates it

    Args:
        path (str): Path to be checked
        create (bool, optional): If path doesn't exists, creates it or not. Defaults to True.

    Returns:
        Bool: Exists or not
    """
    if os.path.exists(path):
        return True
    elif create:
        os.mkdir(path)
        return True
    else:
        return False


def download_image(image_url):
    """Downloads image from an url and returns PIL image

    Args:
        image_url (str): url of the desires image

    Returns:
        PIL Image: downloaded image
    """
    resp = requests.get(image_url, stream=True, timeout=5)
    im_bytes = BytesIO(resp.content)
    image = Image.open(im_bytes)
    return image


def load_string_to_image(image_string):
    image_bytes = base64.urlsafe_b64decode(bytearray(image_string, "utf-8"))
    return Image.open(BytesIO(image_bytes))


def save_image_to_string(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.urlsafe_b64encode(buffered.getvalue())
    return img_str.decode("utf-8")


def draw_label(frame, color, label, timestamp):
    cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), color)
    cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_PLAIN, 1, color)
    cv2.putText(frame, timestamp, (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, color)
    return True


def query_api(url, data):
    response = requests.post(url, data=json.dumps({"image_string": data})).json()
    mask_probability = None
    if response["predictions"]:
        mask_probability = response["predictions"]["mask"]["with_mask"]
    timestamp = response["timestamp"]
    return {"mask_probability": mask_probability, "timestamp": timestamp}


def send_message(client, number_from, number_to, message):
    client.messages.create(
        from_=number_from,
        to=number_to,
        body=message,
    )
    return True
