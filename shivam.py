import face_recognition
import cv2
import numpy as np



print("going")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")
print("")

video_capture = cv2.VideoCapture(0)

shivam_image = face_recognition.load_image_file("shivam_singh.jpg")
shivam_face_encoding = face_recognition.face_encodings(shivam_image)[0]


sameer_image = face_recognition.load_image_file("sameer.jpg")
sameer_face_encoding = face_recognition.face_encodings(sameer_image)[0]


known_face_encodings = [
    shivam_face_encoding,
    sameer_face_encoding
]
known_face_names = [
    "shivam singh",
    "sameer jackrey"
]


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
spinner = 0

while process_this_frame:
   
    ret, frame = video_capture.read()
    #print("frame value")
    #print(frame)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    
    rgb_small_frame = small_frame[:, :, ::-1]

   
    if process_this_frame:
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            #print("here")
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            #print("\033[u")
            #print("Intruder")
            #print("\033[0:0H")
            

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            print("\033[0:0H")
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                print("clear     ")
            else:
                print("intruder")
            face_names.append(name)

    process_this_frame = True

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)
    
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    print("\033[5:0H")
    if spinner == 2:
        spinner = 3
        print("scanning...\\")
    elif spinner == 1:
        spinner = 2
        print("scanning.../")
    elif spinner == 0:
        spinner = 1
        print("scanning...-")
    else:
        spinner = 0
        print("scanning...|")
    print("                    ")

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
