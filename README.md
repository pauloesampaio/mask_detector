# Creating a mask detector and whatsapp alarm system with a raspberry pi camera

I had a [raspberry pi](https://www.raspberrypi.org/) laying around and thought it would be cool to do a face mask detector and alert system!
And it can also be quite useful too. Imagine like Uber or any taxi service. They could use it to monitor the driver and send an alert in case they are driving without a mask. Or even a co-working space, could be an interesting tool to guarantee that protocols are being followed.

<img src="https://paulo-blog-media.s3-sa-east-1.amazonaws.com/posts/2021-02-04-mask_detector/output.gif" alt="">

<img src="https://paulo-blog-media.s3-sa-east-1.amazonaws.com/posts/2021-02-04-mask_detector/twilio.gif" alt="">

## The idea

The architecture can be seen in the schematic below:

<img src="https://paulo-blog-media.s3-sa-east-1.amazonaws.com/posts/2021-02-04-mask_detector/architecture.jpg" alt="">

- Raspberry pi camera opens a video stream
- Each frame of the video is sent to the API container (1)
- The API, running on a container, has 2 steps:
  - Face detection (If there is no face, no use to send to the mask detection)
  - Mask detection
- API answer sent back to the Raspberry pi, that display the frame with the label (2)
- If person is not wearing a mask, Twilio sends a message to the person's whatsapp (or SMS) (3)

## How does it work

### Face detection model

This one is easy. I'm using [OpenCV's DNN face detection model](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector). You just need to:

- download the [model architecture](https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt) and the [model weights](https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180220_uint8/opencv_face_detector_uint8.pb)
- load the model: `model = cv2.dnn.readNetFromTensorflow(model_weights, model_architecture)`
- I'm providing a helper function to made the prediction: [`find_faces(model, image)`](https://github.com/pauloesampaio/mask_detector/blob/f365e6b55c4bd56dfac823d6cdea4baafc3bd076/app/utils/model_utils.py#L119).

This function will give you a list with the bounding boxes of the faces found. If the list is empty, no faces were found in the image.

### Mask detection model

I got from kaggle some mask detection datasets and used them to put together my own, with:

- 4000 with_mask images

<img src="https://paulo-blog-media.s3-sa-east-1.amazonaws.com/posts/2021-02-04-mask_detector/mask.jpg" alt="">

- 4000 without_mask images

<img src="https://paulo-blog-media.s3-sa-east-1.amazonaws.com/posts/2021-02-04-mask_detector/unmasked.jpg" alt="">

Most of this pictures are "in the wild", which helps making the model generalize well for our camera images.
With this dataset I used the Keras implementation of [MobileNetV2](https://keras.io/api/applications/mobilenet/), mainly cause I thought this would be a simple task. Even thought this is a single task, I trained following what I suggested in the [training a multitask neural network](https://pauloesampaio.github.io/posts/2020-12-31-multitask_learning/), just adjusting the config file to account for a single task.

I'm providing the [trained model file](https://paulo-blog-media.s3-sa-east-1.amazonaws.com/posts/2021-02-04-mask_detector/model.h5) and a [training script](https://github.com/pauloesampaio/mask_detector/blob/master/app/train_model.py) that can be ran on Google Colab, in case you don't have a GPU available and want to train it yourself.

For what it's worth, model confusion matrix look really good:
<img src="https://paulo-blog-media.s3-sa-east-1.amazonaws.com/posts/2021-02-04-mask_detector/confusion_matrix.jpg" alt="">

Given a picture with a face on it (as decided by the face detection model), this model predicts if the person is wearing a mask or not.

### Raspberry Pi camera

I'm using [Adrian Rosebrock's imutils](https://github.com/jrosebr1/imutils) library for its optimized picamera videostream. The face detector expects images of 300x300px and the mask detector 224x224px, so having a resolution of 320x240px is more than enough. I'm using an old [2014 5 MP Pi camera v1](https://www.raspberrypi.org/documentation/hardware/camera/) which connects directly to the GPU but, honestly, any USB camera would do the job!

```python
from imutils.video import VideoStream
from imutils.video import FPS
import cv2

vs = VideoStream(usePiCamera=True, 
                 framerate=30, 
                 resolution=(320, 240))
vs.start()
time.sleep(5)
fps = FPS().start()

while True:
    # Get video frame
    frame = vs.read()
    
    # YOU CODE TO DEAL WITH THE FRAMES HERE

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()
```

I'm connecting to the pi over SSH, if you want to check what the camera is getting, forward X11 by using `ssh -X ...`.

### Twilio message

In order to send an alert in case the user is not using a mask, I'm using [Twilio](https://www.twilio.com/). They have a free trial that is more than enough for our test.

```python
from twilio.rest import Client

twilio_client = Client(twilio_account_sid, twilio_auth_token)
client.messages.create(from_=number_from,
                       to=number_to,
                       body=message)
```

Just to be safe and avoid sending too many messages, on the `config.yml` file I added a time delta between messages parameter. So if I'm sending a message now, I'm only sending another message again after `delta` minutes.

## Putting it all together

Now we need to connect these pieces.
I put the face detector and the mask detector in an API running on a Docker container for easy cloud deployment, similar to what I did in the post ["building a prediction API with FastAPI"](https://pauloesampaio.github.io/posts/2021-01-24-prediction_api/). The API was again built using FastAPI, which is quite simple and direct:

```python
# Declares endpoint
@app.post("/predict/") 
async def get_prediction(input_image: InputImage):
    image = load_string_to_image(input_image.image_string)
    # Checks if there's a face on the image. If there is, send it
    # to the mask detector
    faces = find_faces(face_detection_model, image)
    if len(faces) > 0:
        predictions = predict(mask_detection_model, image, config)
        predictions = orjson.loads(
            orjson.dumps(predictions, option=orjson.OPT_SERIALIZE_NUMPY)
        )
    else:
        predictions = None
    
    # Builds a json response with prediction and timestamp
    response = {
        "predictions": predictions,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return response
```

The creators of FastAPI were nice enough to provide a [docker image](https://fastapi.tiangolo.com/deployment/docker/), making the whole process of making it into an optimized and scalable container very easy. I'm providing the [`Dockerfile`](https://github.com/pauloesampaio/mask_detector/blob/master/Dockerfile) and [`docker-compose.yml`](https://github.com/pauloesampaio/mask_detector/blob/master/docker-compose.yml).

Now we can set the whole flow as described on the introduction.

- Raspberry pi opens 320x240 px video stream
- Each frame is encoded in a base64 string
- Each base64 string is sent over the network to the API running on the docker container
- On the API, the image string is decoded back to the PNG format
- Each image goes through the face detection model

  - If no face is detected, API responds with `prediction: None`
  - If there is a face, it goes through the mask detector, which responds with a probability of the person being wearing a mask.

- API response is sent over the network back to the pi
- If the probability is above a given threshold, raspberry pi uses twilio to message the user

All configurations are set on the [`config.yml`](https://github.com/pauloesampaio/mask_detector/blob/master/app/config/config.yml) file, including if you want to visualize the frames and if you run it on the pi or on you local machine's camera (if you don't have a raspberry pi). Some of the main configurations you should look at:

```yaml
  running_on_pi: True # False if you are not running on the pi
  view_frames: True # True to visualize the camera output
  api_url: http://192.168.15.10:8000/predict/ # Network address of where API is running
  message_delta_time_minutes: 30 # How many minutes twilio will wait to send another message
```

There is also the [`credentials.json`](https://github.com/pauloesampaio/mask_detector/blob/master/app/credentials/credentials_example.json) file, where you should put your twilio account details:

```json
    "account_sid": "TWILIO SID",
    "auth_token": "TWILIO AUTH TOKEN",
    "from_number": "YOUR TWILIO NUMBER",
    "to_number": "NUMBER YOU WANT TO SEND MESSAGES TO"
```

I think that's pretty much all, just download the models, put them on the models folder, then start the API by running `docker-compose up` then on another terminal window run `python mask_detector.py`.

## Closing thoughts

I love the Raspberry Pi and I am always thinking about fun ways I can use it. I might add the actual picture in the whatsapp message. Also, in terms of efficiency, probably it would be better to put the messaging service on an API, leaving the pi just to run the video stream.

Hope you have fun with this project and let me know if you have any questions!
