import cv2
import numpy as np 

config = "deploy.prototxt"
model = "dnn_model.caffemodel"

net = cv2.dnn.readNetFromCaffe(config,model)

hat = cv2.imread("hat.png",-1)
glasses = cv2.imread("glasses.png",-1)

print(hat.shape[1])
print(glasses.shape[1])

cap = cv2.VideoCapture(0) 

def OverlayImage(background,overlay,x,y,scale=1):
    print(f"scale",scale)
    overlay = cv2.resize(overlay,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    print(f"overlay shape ",overlay.shape)
    h,w,d = overlay.shape
    for i in range(h):
        for j in range(w):
            if overlay[i,j,3] != 0:
                background[y + i, x + j] = overlay[i, j, :3]
    return background            
    
while True:
    ret,frame = cap.read()
    if not ret:
        break
    h,w = frame.shape[:2]
    
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            # Get the coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")
            x1, y1, x2, y2 = startX, startY, endX, endY

            # Draw the bounding box and confidence score
            label = f"Face: {confidence:.2f}"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            face_width = x2 - x1
            face_height = y2 - y1
            print(f"x1: {x1}, x2: {x2}, y1: {y1}, y2: {y2}")
            
            hat_scale = face_width / hat.shape[1]
            glasses_scale = face_width / glasses.shape[1]
            print(f"Face width: {face_width}, Hat width: {hat.shape[1]}, Glasses width: {glasses.shape[1]}")
            print(f"Hat scale: {hat_scale}, Glasses scale: {glasses_scale}")
            frame = OverlayImage(frame,hat,x1,y1-int(0.5*face_height),hat_scale)
            frame = OverlayImage(frame, glasses, x1, y1 + int(face_height / 4), glasses_scale)


    # Display the frame
    cv2.imshow("Face Detection", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()