SL-LoS-Mock (Will keep updating.....)
=======

### Here is the supporting code for XXXXXXXX.

This repository contains the supporting code for XXXXXXXX.  
There are **five main folders**. To run the code, you first need the **Elucid simulation data**, available at [https://www.elucid-project.com](https://www.elucid-project.com).  

The ray-tracing part was developed by **Chengliang (XXXXXXXXXXXX)**.

---

#### 1. simu2lc
The `simu2lc` folder is used to generate the light cone from simulation data.
To compile and run:

```bash
gcc -lm main.c
./a.out
```
- The output file is specified at **line 248** of `main.c`.  

- **Note:** This file contains the *filling order of subboxes*, not the full light cone.  
  The full light cone includes the dark matter particles.  

- For higher precision, Chengliang split the **N-body 500 Mpc box** into **100 Mpc subboxes** (in this demo).  
  To access these subbox data files, please contact **XXXXXXXX** or **XXXXXXXXX** for collaboration.

#### 2. lc2lensplane
The `lc2lensplane` folder is used to read the simu2lc output file and generate the mass density for each lens plane.
To run it with MPI:

    make, mpirun -np cpu_number ./lrgmlp

Make sure to update your paths at lines 163, 362, and 781.

#### 3. raytracing

1. Run **`cal_lensing_signals.py`** or **`cal_lensing_signals_middel.py`** to calculate:  
   - Deflection angles  
   - Lensing potential  
   - Magnification matrix for each plane  

2. Run **`raytracing_all.py`** to perform ray tracing and obtain the **ray positions on each plane**.

#### 4.CodeMock
This folder contains the code for generating the **final mock data** from the ray-tracing output.

To add **noise** and apply the **PSF** to each image, we use the [**Lenstronomy:** https://lenstronomy.readthedocs.io](https://lenstronomy.readthedocs.io) package.


#### 5. mock_image
The `mock_image` folder contains **100 mock datasets**:

- `image_data.npz` → Lensing images  
- `PointSources.npz` → Point source positions and time delays

##### For questions, collaborations, or requests to use the code, please contact me at **linsj999@outlook.com**.