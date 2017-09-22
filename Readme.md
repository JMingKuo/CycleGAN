# PyTorch style transformer using CycleGAN


## Train-val log
![image](imgs/train-val-log.png)

## Validation Results
**Pastose -> Celluloid** (transformation)
![image](imgs/AB.gif)

**Celluloid -> Pastose** (transformation)
![image](imgs/BA.gif)

**Pastose -> Celluloid -> Pastose** (reconstruction)
![image](imgs/ABA.gif)

**Celluloid -> Pastose -> Celluloid** (reconstruction)
![image](imgs/BAB.gif)


## Notes
- Significant failures in the backgorund during transformation, but the reconstruction results still looks good. 
- Conditional discriminator should helps since there are kind of global color bias obeserved in the generated images, it should be denied easily as long as the discirminator works idealy during model training.

## Reference
1. [https://github.com/sunshineatnoon/Paper-Implementations/tree/master/cycleGAN](https://github.com/sunshineatnoon/Paper-Implementations/tree/master/cycleGAN)
