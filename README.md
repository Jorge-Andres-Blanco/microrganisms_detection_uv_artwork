# microrganisms_detection_uv_artwork

This project attempts to detect colonies of bacteria, fungi, etc... in UV photos of the artwork in the National Theater of Costa Rica.

The first solution for this problem was constructed using a script based on the **OpenCV** function: **matchTemplate** (by Jean Paul Mena Vega in folders JPMV_code)

Considering the stiffness of the first solution, we tried to improve the flexibility of the code by adding rotations to the template and making small, medium, and large versions of them ().

We also attempted to improve the precision of the method by trying to match not only the grayscale template with the image, but also the edges of the template (microorganism colony) with some modifications. 

I am currently trying to solve the problem using a neural network from **YOLOv8** that can be trained from the AGAR Public dataset.

Jorge Andrés Blanco
