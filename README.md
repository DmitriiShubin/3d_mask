## Real-time 3D AR face mask alignment using facial landmarks

### Demo
[![IMAGE ALT TEXT HERE](https://github.com/DmitriiShubin/3d_mask/blob/main/src/preview.png)](https://youtu.be/Fda6uE5K8r0)

### How launch

1. Download the 'data' folder from google drive: https://drive.google.com/file/d/1alSPw5XmCkwBpq0aady02oJuXTRapAIi/view?usp=sharing
2. unzip the archive and move the 'data' folder into the project root "/PATH_TO_REPO/mask"
3. run docker container:
```
sudo docker build . --tag 3d_mask
sudo docker run --name 3d_mask_container -v $(pwd):/ 3d_mask sh /docker-entrypoint.sh
```
### Pipeline

facial landmarks -> calclulation of rotation matrix (theta_x, theta_y, theta_z) -> rotation of the 3D render -> 
calculation of the position -> alignment of the facemask


#### Theory
![alt text](http://url/to/img.png)
#### 1. What does it need to do to align an AR mask on the face?

#### 2. Facial keypoints selection

#### 3. Selection of the axillary keypoints for 3D render

#### 4. Scaling of the mask (imitation of the distance from the camera)

#### 4. Calculation of the position of the mask

#### 4. Calculation of the position of the mask

### Some intermediate results


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