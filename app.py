from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO
from twilio.rest import Client
import cv2
import numpy as np
import pandas as pd
import cvzone
from ultralytics import YOLO


app = Flask(__name__)
socketio = SocketIO(app)

# Initialize Twilio client
account_sid = 'AC72091a9d9a794968d461ce2f39dd771b'
auth_token = '015600450092c8cedfc232f864c0f0e9'
twilio_phone_number = '+16415416411'
your_phone_number = '+916361304142'
client = Client(account_sid, auth_token)

# Initialize YOLO model
model = YOLO('best.pt')

# Load class names
with open("coco1.txt", "r") as f:
    class_list = f.read().splitlines()

def send_sms():
    message = client.messages.create(
        body='An accident has occurred. Please check the location.',
        from_=twilio_phone_number,
        to=your_phone_number
    )
    print("SMS sent:", message.sid)



@app.route('/', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':
        if 'video_file' not in request.files:
            return render_template('index.html')
        
        file = request.files['video_file']
        if file.filename == '':
            return render_template('index.html')
        
        if file:
            # Read uploaded video file
            nparr = np.fromstring(file.read(), np.uint8)
            cap = cv2.VideoCapture('cr.mp4')

            # Load class list from coco1.txt
            my_file = open("coco1.txt", "r")
            data = my_file.read()
            class_list = data.split("\n") 

            count = 0
            while True:    
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                count += 1
                if count % 3 != 0:
                    continue

                rame=cv2.resize(frame,(1020,500))
                results=model.predict(frame)
                #   print(results)
                a=results[0].boxes.data
                px=pd.DataFrame(a).astype("float")
   

                #    print(px)
                for index,row in px.iterrows():
                   x1=int(row[0])
                   y1=int(row[1])
                   x2=int(row[2])
                   y2=int(row[3])
                   d=int(row[5])
                   c=class_list[d]
                if 'accident' in c:
                   cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),1)
                   cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1)
                    # Send SMS when accident is detected
                   send_sms()
                else:
                   cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
                   cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1)
        
                
                cv2.imshow("RGB", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            cap.release()  
            cv2.destroyAllWindows()

            return "Processing complete"
    
    return render_template('index.html')


@socketio.on('connect')
def connect():
    print('Client connected')

@socketio.on('disconnect')
def disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    socketio.run(app, allow_unsafe_werkzeug=True)