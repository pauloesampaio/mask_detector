import cv2
from PIL import Image
from imutils.video import VideoStream
from imutils.video import FPS
from utils.io_utils import (
    yaml_loader,
    save_image_to_string,
    draw_label,
    json_loader,
    send_message,
    query_api,
)
import time
from twilio.rest import Client
from datetime import datetime

config = yaml_loader("./config/config.yml")
credentials = json_loader("./credentials/credentials.json")
message_count = 0
last_message_time = datetime(1, 1, 1)
message_delta_time = 0
twilio_client = Client(credentials["account_sid"], credentials["auth_token"])

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
    # Get video frame
    frame = vs.read()

    # Pass to base64 string
    image_string = save_image_to_string(
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    )

    # send to API
    response = query_api(config["detector"]["api_url"], image_string)

    # If mask probability above threshold ok, else send message
    if response["mask_probability"] > config["detector"]["detector_threshold"]:
        color = (0, 255, 0)
        label = "with mask"
        probability = response["mask_probability"]
    else:
        color = (0, 0, 255)
        label = "without mask"
        probability = 1 - response["mask_probability"]
        message_delta_time = (datetime.now() - last_message_time).seconds / 60

        # If last message was more than delta minutes ago, send message
        if message_delta_time > config["detector"]["message_delta_time_minutes"]:
            send_message(
                twilio_client,
                credentials["from_number"],
                credentials["to_number"],
                config["detector"]["message"],
            )
            message_count = message_count + 1
            last_message_time = datetime.now()

    # Write label on frame
    draw_label(frame, color, f"{label} ({probability:.2f})", response["timestamp"])

    # If you want to check the frames, show them on screen
    if config["detector"]["view_frames"]:
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
# If you are seeing the frames, pressing any key will stop the detector
if config["detector"]["view_frames"]:
    cv2.destroyAllWindows()
vs.stop()
