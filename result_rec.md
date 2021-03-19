
- experiments result record on **CUB**

| model    | image_size       | batch_size | accuracy  | epoch | dataset | date | id     |
| -------- | ---------------- | ---------- | --------- | ----- | ------- | ---- | ------ |
| API-Net  | 600x600(448x448) | 12(12x1)   | 68.36%    | -     | full    | 3.17 |        |
| API-Net  | 600x600(448x448) | 6(6x1)     | 63.68%    | -     | full    | 3.17 |        |
| API-Net  | 256x256(224x224) | 54(18x3)   | 78.56%    | 224   | full    | 3.19 |        |
| resnet50 | 256x256(224x224) | 80         | 77.1%     | 50    | full    | 3.19 | 4698b0 |
| resnet50 | 600x600(448x448) | 10         | ==83.4%== | 50    | full    | 3.18 |        |


>version -1 6x1 Stage1: ResNet50 68.48% 这里的baseline没有达到84.5
>version       ResNet50        dev     74.8%
>version       ResNet50        dev     70.0%   添加dropout0.5
>version       ResNet50                67.03%
>version       ResNet50        dev     73.4%
>version 1.0 6x1  57%     3/13