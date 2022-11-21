# GAZE3D
Analysis toolbox for 3d tracking and gaze data, designed mainly for VR tracking data.

## Input data

Eye tracking:
-Left eye origin
-Right eye origin
-Left eye gaze vector
-Right eye gaze vector

Intersection with environment
-Gaze hitpoint
-Hitpoint object name

Head tracking
-Camera/head origin
-Camera/head quaternion




## Functions
### Head-relative gaze to world gaze
gaze2world()
### Cartesian gaze vec to spherical coordinates
### Calculate vergence distance
vergenceDist()
### Yaw pitch and roll from quaternion
quat2ypr()

<img src="/docs/yaw_pitch_roll.png" width="300">

Independent of the defined orientation of the coordinate system
- yaw is the rotation around the vertical axis
- pitch is the rotation around the transverse (lateral) axis
- roll is the rotation around the longitodinal (forward) axis

We follow the convention of yaw, pitch and roll as the [Tait-Bryan angles of an intrinsic rotation with the order vertical axis - transverse axis - roll axis](https://www.mauriciopoppe.com/notes/computer-graphics/transformation-matrices/rotation/euler-angles/)
. Based on quaternions measured in Unity, the coordinate system is left-handed with the
- vertical axis y
- forward axis z
- transverse axis x.
Therefore, the Tait-Bryan angles used for yaw, pitch and roll are y - x' - z'' (z-x-y as extrinsic rotation).
