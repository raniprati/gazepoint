## gazepoint
Find and show gaze point from live video stream

Run code file: estimategaze.py : to capture video and display Gaze point coordinate on the video screen and print related code values on terminal.

#Method used for Gaze point estimation
Single Eye based Gaze Point estimation as discussed in reference paper (3), using Polynomial Affine Transformation (4) (5) (8) (9) as discussed in section 3.2 of reference paper (1) and data transformation methods discussed in section 4 of reference paper (2) to implement the Pinhole camera model (12).
Mediapipe Facemesh model (10)(11) is used for face landmark points.
Model and Camera calibaration approximation values are used from reference (6).

#References
(1) https://www.smohanty.org/Publications_Journals/2019/Mohanty_IEEE-Potentials_2019-Jan_Gaze-Estimation.pdf
(2) https://arxiv.org/pdf/2104.12668.pdf
(3) https://escholarship.org/content/qt6pb6q6gt/qt6pb6q6gt_noSplash_f60bc039a2de196a616d765811897a43.pdf
(4) https://docs.opencv.org/4.5.4/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d
(5) https://amroamroamro.github.io/mexopencv/matlab/cv.estimateAffine3D.html
(6) https://www.kaggle.com/code/dasmehdixtr/nose-pose-estimation-via-opencv/notebook 
(7) https://towardsdatascience.com/image-geometric-transformation-in-numpy-and-opencv-936f5cd1d315
(8) https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga396afb6411b30770e56ab69548724715
(9) https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga1019495a2c8d1743ed5cc23fa0daff8c
(10) https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py
(11) https://github.com/google/mediapipe/blob/master/mediapipe/tasks/python/vision/face_landmarker.py
(12) https://en.wikipedia.org/wiki/Pinhole_camera_model
(13) https://medium.com/mlearning-ai/eye-gaze-estimation-using-a-webcam-in-100-lines-of-code-570d4683fe23
(14) https://kh-monib.medium.com/title-gaze-tracking-with-opencv-and-mediapipe-318ac0c9c2c3
