
- experiments result record on **CUB**

| model                               | image_size       | batch_size | accuracy  | epoch | dataset | date | id     |
| ----------------------------------- | ---------------- | ---------- | --------- | ----- | ------- | ---- | ------ |
| API-Net                             | 600x600(448x448) | 12(12x1)   | 68.36%    | -     | full    | 3.17 |        |
| API-Net                             | 600x600(448x448) | 6(6x2)     | 84.1%     | 204   | full    | 3.17 |        |
| API-Net                             | 256x256(224x224) | 54(18x3)   | 78.65%    | 224   | full    | 3.19 |        |
| **resnet50**                        | 256x256(224x224) | 80         | **77.5%** | 50    | full    | 3.19 | 4698b0 |
| resnet50                            | 600x600(448x448) | 10         | ==83.4%== | 50    | full    | 3.18 |        |
| resnet50 + model1:cosine_similarity | 256x256(224x224) | 40         | 80.1%     | 90    | full    | 3.21 |        |
| resnet50 + model1:L1Loss            | 256x256(224x224) | 40         | 79.1%     | 100   | full    | 3.22 |        |
| resnet50 + model1:MSELoss           | 256x256(224x224) | 40         | 78.7%     | 90    | full    | 3.22 |        |
| resnet50 + model1:project+cosine    | 256x256(224x224) | 40         | 80.0%     | 85    | full    | 3.22 |        |
| resnet50 + model1:project+L1Loss    | 256x256(224x224) | 40         | 79.5%     | 50    | full    | 3.22 |        |
| resnet50 + model1:project+MSE       | 256x256(224x224) | 40         | --        | --    | full    | 3.22 |        |

- experiments result on part dataset

| model                               | image_size       | batch_size | accuracy | epoch | dataset | date | id  |
| ----------------------------------- | ---------------- | ---------- | -------- | ----- | ------- | ---- | --- |
| resnet50                            | 256x256(224x224) | 80         | 88.8%    | 50    | part    | 3.19 |     |
| resnet50 + model1:cosine_similarity | 256x256(224x224) | 40         | 90%      | 30    | part    | 3.21 |     |
