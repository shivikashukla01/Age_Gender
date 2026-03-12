import cv2
import torch
from PIL import Image
from torchvision import transforms

from model import AgeGenderModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = AgeGenderModel().to(device)

model.load_state_dict(torch.load("models/age_gender_model.pth", map_location=device))

model.eval()


face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray,1.3,5)


    for (x,y,w,h) in faces:

        face = frame[y:y+h, x:x+w]

        face_rgb = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)

        face_pil = Image.fromarray(face_rgb)

        face_tensor = transform(face_pil).unsqueeze(0).to(device)


        with torch.no_grad():

            age_pred, gender_pred = model(face_tensor)

            age = int(age_pred.item())

            gender = torch.argmax(gender_pred).item()

            gender = "Male" if gender==0 else "Female"


        label = f"{gender}, {age}"


        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.putText(frame,label,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,(0,255,0),2)


    cv2.imshow("Age Gender Prediction",frame)

    if cv2.waitKey(1)==27:
        break


cap.release()
cv2.destroyAllWindows()