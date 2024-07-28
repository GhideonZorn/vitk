# VTK/ITK Project

## Project link

https://github.com/GhideonZorn/vitk

## Contributors

- [Ghislain Bonnard](ghislain.bonnard@epita.fr)
- [Jules Wiriath](jules.wiriath@epita.fr)
- [Anaïs Raison](anais.raison@epita.fr)

## Dependencies installation

`mkdir venv`
`python3 -m venv ./venv`
`source ./venv/bin/activate`
`pip3 install -r requirements.txt`

## Usage

```bash
python main.py
```

The main will open a dash application. The user can launch the registration, then the
segmentation and visualise them in 2D or 3D by clicking the two different visualisation
buttons.

- Registration : launchs preprocessing and recalibration
- Visualisation (upper) : plots several 2D images
- Segmentation : launchs the segmentation
- Visualisation (lower) : launchs the 3D visualisation
- Run everything : luanhcs registration, segmentation and 2D and 3D visualisations

## Solution explanation

### Recalibration

We tested two preprocessing approaches : one using a median filter to reduce noise and another using a curvature flow filter. In the end, we chose to keep the first one because the second was too aggressive, and the segmentation results were not convincing.

The registration aligns the moving image to the fixed image through a translation transform, measuring similarity with the Mean Squares Image to Image Metric, and optimizing with a Regular Step Gradient Descent Optimizer. This approach ensures accurate alignment by addressing noise, maintaining consistent image dimensions, and utilizing a similarity metric and optimization method. Consequently, the transformed image aligns correctly with the fixed image, facilitating precise segmentation and analysis.

### Segmentation

We chose a threshold approach for segmentation using lower and upper boundaries provided to the function. This threshold is applied in a region growing algorithm that starts from two given seeds. These seeds are placed on the two tumors in the brain.

Without using seeds, the threshold would retain not only the tumors but also some noise. By starting from the two tumors and only keeping pixels between the lower and upper boundaries, we achieved excellent results.

### Visualisation

We have a 2D visualisation of different images : 
  - moving image
  - fixed image
  - transformed image
  - segmented image

We also chose to have a 3D visualization with the head cut in half and the tumors in red. This provides a view of the entire tumors in 3D, and by using the cut head, the location of the tumors in the brain can be visualized.
