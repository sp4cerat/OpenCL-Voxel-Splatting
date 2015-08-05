# OpenCL-Voxel-Splatting
Splatting using OpenCL can reach more than 2 Bln Splats/s !

In response to rumors around unlimited detail, I wanted to give point based rendering at try and see how well it works in OpenCL.

It turned out that the performance is much better than basic point rendering in OpenGL. While OpenGL allowed me to render around 630M pts / sec, OpenCL reached 3-4 times the speed with ~2 Billion points per second on a GTX580M GPU.

Just rendering points does not lead to a smooth surface however. For that, a post-processing filter is required. It increases the size of the points and fills the holes. I have implemented a very simple one, to show that it works. As for the culling, only frustum culling is implemented. More advanced hierarchic occlusion culling might give extra frames especially in indoor scenes. Also a hierarchic depth buffer is not been used - it might further give additional performance when adapted to point based rendering.

Main challenge for the implementation was to find a data structure that allows parallel access by stil keeping the size per voxel reasonable. Rendering the points works pretty much straight forward.

In the image, you can see the render stages:
Right : Z buffer
Middle : Colored Points
Left	: Including post processing

Details:

Render Resolution: 1024x768
Framerate: 30-40 fps

Scene Dimension: 20k x 1k x 20k voxels
Dataset : 1024x1024x1024 Voxels (single instance)
Data size per voxel: 4 bytes / voxel at each LOD, total ~5 bytes, seen over all LODs


[![Screenshot1](https://github.com/sp4cerat/OpenCL-Voxel-Splatting/blob/master/screenshot.jpg?raw=true)](https://www.youtube.com/watch?v=CyyhWkMmgeE)
