## Real-time 3D AR face mask alignment using facial landmarks
The following repo describes the pipeline of real-time AR 3D face mask alignment 

Availiable facemasks:
1. FrontMan, Squid Games TV series (https://youtu.be/azCmCA4WhrI)
2. Ram (https://youtu.be/mltJyCFM2Xc) (https://youtu.be/G33-h3I2sY8)
3. Mastache (Probably, the least interesting)
4. Joker (https://youtu.be/MmZk9iWFJcU)

**The Open3D rendering engine was replaced with Pyvista because it allows using of textures. However, Pyvista has internal memory leak since 2019. Therefore, the app runs only ~ 1 minute on my 16Gb RAM laptop before crushing.**

### Demo
[![IMAGE ALT TEXT HERE](https://github.com/DmitriiShubin/3d_mask/blob/main/src/preview.jpg)](https://youtu.be/azCmCA4WhrI)

### How launch

1. Download the 'data' folder from google drive: https://drive.google.com/file/d/1alSPw5XmCkwBpq0aady02oJuXTRapAIi/view?usp=sharing
2. unzip the archive and move the "data" folder into the project root "/PATH/mask/"
3. install python packages for your virtual environment:
```
pip install -r requirements.txt
jupyter nbextension enable --py --sys-prefix ipygany
```
4. Run command from root:
```
python demo.py
```

You can also select alailiable masks at "./config.yaml"

### Pipeline
![alt text](https://github.com/DmitriiShubin/3d_mask/blob/main/src/pipeline.jpg)


The overall pipeline includes the following steps:

1. Detection of facial landmarks (I used mediapipe)
2. Calculation of features based on facial landmarks
3. Estimation of the head pose using extracted features
4. Applying rotation on the 3D model
5. Rendering the 3D model into a 2D image
6. Scaling of the mask image using projections of model's points and distance between eyes (landmarks extracted from the face)
7. Location of the mask position using projections of model's points and eye's positions
8. Applying binary mask on mask's image
9. Alignment of the mask and face images

Additional details:

I used a mean average filter on mask scaling, location, and rotation to make everything smoother.

#### 3D mask face alignment

In general, the alignment of the projection of the 3D object into a 2D image is the rendering problem. Ideally, we have to know the exact position of the head (rotation, distance from the camera, location concerning the camera) to apply the translations on the 3D object and obtain the 2D render of that object. However, this approach has several issues:

We need an accurate single-camera depth map to compute the normal face surface to estimate the distance and head pose. Of course, the domain of the problem is relatively close compared to the generic monocular depth map estimation problem (we need to get the depth map of only the face), which makes the model a bit easier to fit. But since the projection of the 3D object into the camera has a significant level of ambiguity of perspective, this pipeline will not be super robust. In addition, it needs to know the parameters of cameras' optical distortions, which makes the problem complex in terms of generalization for various cameras.

Depth map estimation requires encoder-decoder NNs architectures, which creates additional computation issues for real-time processing.

How can we avoid those issues?

Let's summarise several assumptions that we know about the human face:

1. Human face has a property of vertical symmetry.
2. The upper part of the face (eyes and forehead) is stationary, reflects the face's position, and is irrelevant to facial expressions.
3. Face proportions vary a bit for different people, but in general, they provide a certain degree of generalization.
4. Since the user always looks at the camera, the estimated degree angle is limited by -+ 30 degrees.

Sounds better, huh?

Another critical assumption is that if we estimate the head pose (3 angles of rotation) correctly and adequately orient the 3D mask model, the only thing we need to do is to scale the mask to the size of the face and locate it on the face front.

We can do it easily using eye landmarks: 
![alt text](https://github.com/DmitriiShubin/3d_mask/blob/main/src/landmarks.jpg)

To do that, I created additional axillary points on the 3D model near eyes that will be rotated with the 3D model of the face mask. When the rotation is applied, projections of reference points will be obtained for the final fitting of the mask on the face.


#### Head pose estimation using face keypoints

##### Z rotation
Let's start with the simples one. Since the model I used for eyes landmarks detection is invariant to head rotation, the Z-axis head rotation could be computed explicitly using the Right triangle rule:
![alt text](https://github.com/DmitriiShubin/3d_mask/blob/main/src/z_rotation.jpg)

```
def calculate_angle_x(self, left_eye: Tuple[int, int], right_eye: Tuple[int, int]) -> float:

    x = np.array(right_eye)
    y = np.array(left_eye)

    proj = x[1] - y[1]

    if proj > 0:
        pos = True
    elif proj < 0:
        pos = False
    else:
        return 0

    proj = np.abs(proj)

    ct = 1 / np.tan(proj / (x[0] - y[0]))

    if pos:
        return 90 - np.degrees(np.arctan(ct))
    else:
        return np.degrees(np.arctan(ct)) - 90
```

##### Y rotation
The calculation of Y is a bit more complex. The angle of Y-axis rotation is defined by the ratio of distances forehead-left_eye and forehead-right_eye. To estimate this relation, I trained a linear model on facial landmarks extracted from [Biwi Kinect Head Pose Dataset](https://www.kaggle.com/kmader/biwi-kinect-head-pose-database).
![alt text](https://github.com/DmitriiShubin/3d_mask/blob/main/src/y_rotation.jpg)

```
a = self._compute_distance_between_points(left_eye, forehead)
b = self._compute_distance_between_points(right_eye, forehead)
features['upper_sides_proportion'] = a / b
```

##### X rotation
The X-axis is the most difficult to compute. Again, I used the same Biwi dataset, using a ratio of the distances forehead-mid eye and left eye-right eye as a feature for calculating the angle.
![alt text](https://github.com/DmitriiShubin/3d_mask/blob/main/src/x_rotation.jpg)

```
c = self._compute_distance_between_points(right_eye, left_eye)
d = self._compute_distance_between_points(forehead, center)
features['vertical_sides_proportion'] = d / c
```
