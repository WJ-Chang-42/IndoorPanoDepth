# IndoorPanoDepth
We present a novel neural representation based method for depth estimation from a few panoramic images of different views. 
![](./fig/teaser.png)
This is the official repo for the implementation of **Depth Estimation from Indoor Panoramas with Neural Scene Representation**.

## Usage
### Matterport3D
Direct run the following command.
```
sh Matterport3D.sh
```
---
### Stanford2D3D
Direct run the following command.
```
sh Stanford2D3D.sh
```
---
### Our dataset
First download our dataset from https://anonfiles.com/8cx3ffI7y9/ours_zip. Then, unzip it to the './data' fold as follows:
```
|-- code
    |-- data      
        |-- Matterport3D
        |-- Stanford2D3D
        |-- ours 
            |-- bedroom  
            ...
```
Finally, run the command
```
sh ours.sh
```


## Acknowledgement
The main framework is borrowed from [NeuS](https://github.com/Totoro97/NeuS). The 3D models used for rendering dataset are from [Flavio, Della, Tommasa](https://download.blender.org/demo/cycles/flat-archiviz.blend), [Christophe, Seux](https://download.blender.org/demo/test/classroom.zip) and [Tadeusz](https://blenderartists.org/t/free-scene-loft-interior-design/1200857).
Thanks for these great works.