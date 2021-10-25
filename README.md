## Real-time 3D AR face mask alignment using facial landmarks

### Demo
[![IMAGE ALT TEXT HERE](https://github.com/DmitriiShubin/3d_mask/blob/main/src/preview.png)](https://youtu.be/Fda6uE5K8r0)

### How launch

1. Download the 'data' folder from google drive: https://drive.google.com/file/d/1alSPw5XmCkwBpq0aady02oJuXTRapAIi/view?usp=sharing
2. unzip the archive and move the "data" folder into the project root "/PATH/mask/"
3. install python packages for your virtual environment:
```
pip install -r requirements.txt
```
4. Run command from root:
```
python demo.py
```

### Pipeline
![alt text](https://github.com/DmitriiShubin/3d_mask/blob/main/src/pipeline.jpg)


The overall pipeline includes following steps:
1. Detection of facial landmarks (I used mediapipe)
2. Calculation of features based on facial landmakrs
3. Estimation of the head pose using extracted features
4. Applying rotation on the 3D model
5. Rendering the 3D model into 2D image
6. Scaling of the mask image using projections of model's points and distance between eyes (landmarks extracted from the face)
7. Location of the mask position using projections of model's points and eye's positions
8. Applying binary mask on mask's image
9. Alignment of the mask and face images

Additional details:
- I used a mean average filter on mask scaling, location and rotation to make everything smoother

#### 3D mask face alignment

In general, the alignment of the projection of the 3D object into a 2D image is the rendering problem. Ideally, we have to know the exact position of the head (rotation, distance from the camera, location concerning the camera) to apply the translations on the 3D object and obtain the 2D render of that object. However, this approach has several issues:

1. We need an accurate single-camera depth map to compute the normal face surface to estimate the distance and head pose. Of course, the domain of the problem is relatively close compared to the generic monocular depth map estimation problem (we need to get the depth map of only the face), which makes the model a bit easier to fit. But since the projection of the 3D object into the camera has a significant level of ambiguity of perspective, this pipeline will not be super robust. In addition, it needs to know the parameters of cameras' optical distortions, which makes the problem complex in terms of generalization for various cameras.

2. Depthmap estimation requires encoder-decoder NNs architectures, which creates additional computation issues for real-time processing.

How can we avoid those issues?
 
Let's summarise several assumptions that we know about the human face:

1. In general, the face has a property of vertical symmetry. 
2. The upper part of the face (eyes and forehead) is stationary, reflects the face's position, and is irrelevant to facial expressions. 
3. SInce the user is always looking at the camera, the estimated degree angle is limited by -+ 30 degrees. 

Sounds better, huh?

Another critical assumption is that if we estimate the head pose (3 angles of rotation) correctly and adequately orient the 3D mask model, the only thing we need to do is to scale the mask to the size of the face and locate it on the face front.

We can do it easily using eye landmarks:
![alt text](https://github.com/DmitriiShubin/3d_mask/blob/main/src/landmaks.jpg)

To do that, I created additional axillary points on the 3D model near eyes that will be rotated with the 3D model of the face mask. When the rotation is applied, projections of reference points will be obtained for the final fitting of the mask on the face.


#### Head pose estimation using face keypoints



### Some intermediate results

The general framework of alignment of the AR 3D mask implies the rendering problem when the arbitrary 3D object has to be rendered into the 2D image (projection).

Ideally, it requires the following known parameters:
1. relative position of the face with respect to the camera position
2. estimation of the head pose with respect to the camera

According to this, the pipeline should look like:
1. Estimation of the depth (monocular, using one camera)
2. Calclulating the normal vector for the face surface
3. Estimatiopn of the pose using face normal
4. Adjusting the camera parameters and rendering to the image.

However, this approach has 

How can we simplify it?

There are several assumptions we can take:
1. Our domain is closed to the human face, i.e. the 
2. Symmetry
3. Human faces are sharing the similarity
3. angles 

Ho


### What didn't work
1. detection of the face => adding keypoints 
20-> 30 FPS
probably could be beneficial when training and compression of the separate landmark detection model

2. 



TODO:
3. add description 
4. Check up how it runs on mac

record a demo
add docker i