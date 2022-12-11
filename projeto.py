import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


cap = cv2.VideoCapture(0)
success, image = cap.read()


height, width, _ = image.shape

masked = np.zeros((height, width, 3), np.uint8)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue
    image = cv2.flip(image, 1)
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    x_min, x_max, y_min, y_max = 480, 0, 360, 0
    
    
    hull_list = []
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        #mp_drawing.draw_landmarks(
        #    image,
        #    hand_landmarks,
        #    mp_hands.HAND_CONNECTIONS,
        #    mp_drawing_styles.get_default_hand_landmarks_style(),
        #    mp_drawing_styles.get_default_hand_connections_style())
        
        for i in range(0, len(hand_landmarks.landmark)):
            pt = hand_landmarks.landmark[i]
            hull_list.append((int(pt.x*width), int(pt.y*height)))
            #cv2.circle(image, (int(pt.x * width), int(pt.y * height)), 5, (0,0,255))
    
    
    if hull_list is not None and len(hull_list) > 0:
        hull_np = np.array(hull_list)
        
        masked = np.zeros(image.shape[:2], np.uint8)
        
        convexhull = cv2.convexHull(hull_np)
        
        
        #cv2.polylines(masked, [convexhull], True, (255,255, 255), 3)
        cv2.fillConvexPoly(masked, convexhull, (255, 255, 255))
        cv2.polylines(image, [convexhull], True, (0, 255, 0))
        
        image = cv2.resize(image, (width, height))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        masked = cv2.resize(masked, (width, height))

        masked_image = cv2.bitwise_and(image, image, mask = masked)
        
        image = cv2.bitwise_and(gray, gray, mask = masked)
        
        gray = np.stack((gray,)*3, axis = -1)
        
        added_img = cv2.add(masked_image, gray)
        
        cv2.imshow('added', added_img)

    cv2.imshow('MediaPipe Hands', image)
    
    
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
cv2.destroyAllWindows()