# IACV_Projective_Synchronization

The goal of synchronization is to estimate the values of some unknown states related to a network, given some measurements, typically corrupted by noise, between pairs of nodes.

## Image Mosaicing

Synchronization can be used to solve the image mosaicing problem, whose goal, given a set of images from which pairs of relative homographies are computed, is to align the images into a single mosaic.

![image](https://github.com/CarloSgaravatti/IACV_Projective_Synchronization/assets/58942793/1d0f3e8a-690f-4737-b83a-5594b935c71b)

## Projective Synchronization

This project extends some techniques used to solve the image mosaicing problem to the projective synchronization problem, in which the goal is to align different projective frames into a single global projective frame.

## Documentation

Details on the techniques used in this project can be found [here](docs/report.pdf)

## References

<a id="1">[1]</a> 
F. Arrigoni and A. Fusiello. 
Synchronization problems in computer vision with closed-form solutions. 
International Journal of Computer Vision, 128, 01 2020.

<a id="2">[2]</a> 
E. Santellani, E. Maset, and A. Fusiello.
Seamless image mosaicking via synchronization.
ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences, IV-2:247–254, 2018.

<a id="3">[3]</a> 
P. Schroeder, A. Bartoli, P. Georgel, and N. Navab.
Closed-form solutions to multiple-view homography estimation.
In 2011 IEEE Workshop on Applications of Computer Vision (WACV), pages 650–657, 2011.

<a id="4">[4]</a> 
L. Yang, H. Li, J. A. Rahim, Z. Cui, and P. Tan.
End-to-end rotation averaging with multi-source propagation.
In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11769–11778, 2021.
