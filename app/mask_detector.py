import cv2
import requests
import json
from PIL import Image
from imutils.video import VideoStream
from imutils.video import FPS
from utils.io_utils import yaml_loader, save_image_to_string, draw_label, json_loader
import time
from twilio.rest import Client
from datetime import datetime

config = yaml_loader("./config/config.yml")
credentials = json_loader("./credentials/credentials.json")
message_count = 0
last_message_time = datetime(1, 1, 1)
message_delta_time = 0
client = Client(credentials["account_sid"], credentials["auth_token"])

if config["detector"]["running_on_pi"]:
    vs = VideoStream(
        usePiCamera=True, framerate=30, resolution=config["detector"]["resolution"]
    )
    vs.start()
else:
    vs = VideoStream(usePiCamera=False, framerate=30).start()
    vs.stream.set(cv2.CAP_PROP_FRAME_WIDTH, config["detector"]["resolution"][0])
    vs.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, config["detector"]["resolution"][1])


time.sleep(5)
fps = FPS().start()

while True:
    frame = vs.read()
    image_string = save_image_to_string(
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    )
    response = requests.post(
        config["detector"]["api_url"],
        data=json.dumps({"image_string": image_string}),
    ).json()
    if response["predictions"]["mask"]["with_mask"] > 0.5:
        color = (0, 255, 0)
        label = "with mask"
        probability = response["predictions"]["mask"]["with_mask"]
    else:
        color = (0, 0, 255)
        label = "without mask"
        probability = response["predictions"]["mask"]["without_mask"]
        message_delta_time = (datetime.now() - last_message_time).seconds / 60
        print(message_delta_time)
        if (message_count == 0) or (
            message_delta_time > config["detector"]["message_delta_time_minutes"]
        ):
            message = client.messages.create(
                to=credentials["to_number"],
                body="Coloque a máscara!!!",
                from_=credentials["from_number"],
            )
            print("OI")
            print("Message sent")
            message_count = message_count + 1
            last_message_time = datetime.now()

    draw_label(frame, color, f"{label} ({probability:.2f})", response["timestamp"])
    if config["detector"]["view_frames"]:
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
if config["detector"]["view_frames"]:
    cv2.destroyAllWindows()
vs.stop()
