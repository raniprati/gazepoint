
import mediapipe as mp
import cv2
import numpy as np

# Convert normalized Facemesh model points (present  in normalized range [-1,1]) back to original values relative to the frame size
denormalize2D = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]))
denormalize3D = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]), 0)

def find_gazepoint(imgframe, points, scrdist):
    image_points2D = np.array([
    denormalize2D(points.landmark[4], imgframe.shape),  #Face Landmark Nose tip
    denormalize2D(points.landmark[152], imgframe.shape),  #Face Landmark Chin
    denormalize2D(points.landmark[263], imgframe.shape),  #Face Landmark Left Eye Left corner
    denormalize2D(points.landmark[33], imgframe.shape),  #Face Landmark Right Eye Right corner
    denormalize2D(points.landmark[287], imgframe.shape),  #Face Landmark Left Mouth corner
    denormalize2D(points.landmark[57], imgframe.shape)  #Face Landmark Right Mouth corner
    ], dtype="double")

    image_points3D = np.array([
        denormalize3D(points.landmark[4], imgframe.shape),  #Face Landmark Nose tip
        denormalize3D(points.landmark[152], imgframe.shape),  #Face Landmark Chin
        denormalize3D(points.landmark[263], imgframe.shape),  #Face Landmark Left Eye, Left Corner
        denormalize3D(points.landmark[33], imgframe.shape),  #Face Landmark Right Eye Right Corner
        denormalize3D(points.landmark[287], imgframe.shape),  #Face Landmark Left Mouth Corner
        denormalize3D(points.landmark[57], imgframe.shape)  #Face Landmark Right Mouth Corner
    ], dtype="double")

    # 3D FaceMesh Model Calibaration Points based on Nose Tip as Origin to be used for Affine transformation
    model_points3D = np.array([
        (0.0, 0.0, 0.0),  # Nose Tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye, left corner
        (225.0, 170.0, -135.0),  # Right eye, right corner
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

    # 3D FaceMesh Model Calibaration Points based on Nose Tip as Origin for the Center of the Right Eyeball
    # taken to be on the line of Right Mouth
    Eyeball_center_right = np.array([[150.0], [170.0], [-135.0]])


    # Camera Matrix Calibaration approximated on the basis of Image Frame Size
    focal_length = imgframe.shape[1]
    center = (imgframe.shape[1] / 2, imgframe.shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    camera_dist_coeffs = np.zeros((4, 1))  # Camera Dist coeffs approximation assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points3D, image_points2D, camera_matrix,
                                                                  camera_dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # 2D pupil locations from FaceMesh Model
    rightpupil = denormalize2D(points.landmark[469], imgframe.shape)
    leftpupil = denormalize2D(points.landmark[474], imgframe.shape)
    print("Left pupil of user: ", leftpupil)
    print("Right pupil of user: ", rightpupil)

    #Transformation of 3D frame image points to 3D Real World Points
    _, transformation, _ = cv2.estimateAffine3D(image_points3D, model_points3D)


    if transformation is not None:  # if estimateAffine3D is successful
        # Project pupil frame image point to 3D Real World Points
        rightpupil_cord_model3d = transformation @ np.array([[rightpupil[0], rightpupil[1], 0, 1]]).T
        print("3D Right pupil coordinates of user: ", rightpupil_cord_model3d)

        # 3D gaze points of Right Eye Pupil on Screen with provided screen distance
        Sright = Eyeball_center_right + (rightpupil_cord_model3d - Eyeball_center_right) * scrdist

        # Project 3D gaze points of Left Eye Pupils to 2D image plane
        (rightgaze2D, _) = cv2.projectPoints((int(Sright[0]), int(Sright[1]), int(Sright[2])), rotation_vector,
                                             translation_vector, camera_matrix, camera_dist_coeffs)
        print("2D right gaze point of user:: ", rightgaze2D)

        # Project 3D head pose into the 2D image plane with provided screen distance
        (headpose_wrt_right, _) = cv2.projectPoints((int(rightpupil_cord_model3d[0]), int(rightpupil_cord_model3d[1]), int(scrdist)),
                                                    rotation_vector, translation_vector, camera_matrix, camera_dist_coeffs)
        print("head pose for right pupil of user:: ", headpose_wrt_right)

        # Correct gaze point for head rotation
        gazepoint_right = rightgaze2D[0][0] - (headpose_wrt_right[0][0] - rightpupil)
        print("Head Pose corrected right gaze point of user:: ", gazepoint_right)

        #Display Gaze point and direction on Screen
        pupilleft = (int(leftpupil[0]), int(leftpupil[1]))
        pupilright = (int(rightpupil[0]), int(rightpupil[1]))
        gazepoint_coord = (int(gazepoint_right[0]), int(gazepoint_right[1]))
        print("Coordinates displayed of Gaze point of user:: ", gazepoint_coord)

        cv2.line(imgframe, pupilleft, gazepoint_coord, (0, 0, 255), 2)
        cv2.line(imgframe, pupilright, gazepoint_coord, (0, 0, 255), 2)
        cv2.putText(imgframe, "Gaze Point:  " + str(gazepoint_coord), (70, 130), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1)
    return


def face_capture(screendist):
    mp_face_mesh = mp.solutions.face_mesh
    # camera stream:
    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5 ) as face_mesh:
        while cap.isOpened():
            success, frameimage = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            frameimage.flags.writeable = False
            frameimage = cv2.cvtColor(frameimage, cv2.COLOR_BGR2RGB)  # frame to RGB for the FaceMesh Model
            results = face_mesh.process(frameimage)
            outimage = cv2.cvtColor(frameimage, cv2.COLOR_RGB2BGR)  # frame back to BGR for OpenCV

            if results.multi_face_landmarks:
                find_gazepoint(outimage, results.multi_face_landmarks[0], screendist)

            cv2.imshow('webcam', outimage)
            if cv2.waitKey(3) & 0xFF == ord('q'):
                break
        cap.release()
    return

if __name__ == '__main__':
    screen_distance = 40
    face_capture(screen_distance)