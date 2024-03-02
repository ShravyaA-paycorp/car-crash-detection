import cv2
import pandas as pd
from ultralytics import YOLO
from twilio.rest import Client
import cvzone

# Twilio credentials
account_sid = 'AC72091a9d9a794968d461ce2f39dd771b'
auth_token = 'a10005f0ecd073959b248f7084d20c0e'
twilio_phone_number = '+16415416411'
your_phone_number = '+916361304142'

# Initialize Twilio client
client = Client(account_sid, auth_token)



model=YOLO('best.pt')

def send_sms():
    message = client.messages.create(
        body='An accident has occurred. Please check the location.',
        from_=twilio_phone_number,
        to=your_phone_number
    )
    print("SMS sent:", message.sid)


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
            
        



cap=cv2.VideoCapture('car.mp4')


my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)



count=0

while True:    
    ret,frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
   

    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
   

#    print(px)
    for index,row in px.iterrows():
#        print(row)
 
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
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()  
cv2.destroyAllWindows()




