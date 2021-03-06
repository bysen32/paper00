
- experiments result record on **CUB**

| model                                        | image_size       | batch_size | accuracy  | epoch | dataset | date | id     |
| -------------------------------------------- | ---------------- | ---------- | --------- | ----- | ------- | ---- | ------ |
| API-Net                                      | 600x600(448x448) | 12(12x1)   | 68.36%    | -     | full    | 3.17 |        |
| API-Net                                      | 600x600(448x448) | 6(6x2)     | 84.1%     | 204   | full    | 3.17 |        |
| resnet50                                     | 600x600(448x448) | 10         | ==83.4%== | 50    | full    | 3.18 |        |
| API-Net                                      | 256x256(224x224) | 54(18x3)   | 78.65%    | 224   | full    | 3.19 |        |
| **resnet50**                                 | 256x256(224x224) | 80         | **77.5%** | 50    | full    | 3.19 | 4698b0 |
| resnet50 + model1:cosine_similarity          | 256x256(224x224) | 40         | 80.1%     | 90    | full    | 3.21 |        |
| resnet50 + model1:L1Loss                     | 256x256(224x224) | 40         | 79.1%     | 100   | full    | 3.22 |        |
| resnet50 + model1:MSELoss                    | 256x256(224x224) | 40         | 78.7%     | 90    | full    | 3.22 |        |
| resnet50 + model1:project+cosine             | 256x256(224x224) | 40         | 80.0%     | 90    | full    | 3.22 |        |
| resnet50 + model1:project+100*cosine         | 256x256(224x224) | 40         | 81.0%     | 200   | full    | 3.23 |        |
| resnet50 + model1:project+L1Loss             | 256x256(224x224) | 40         | 79.5%     | 50    | full    | 3.22 |        |
| resnet50 + model1:project+MSE                | 256x256(224x224) | 40         | --        | --    | full    | 3.22 |        |
| resnet50 + model1:p+c + model2               | 256x256(224x224) | 20         | 77.7%     | 500   | full    | 3.24 |        |
| resnet50 + model1:pp + model2:ppp1 + triplet | 256x256(224x224) | 13x3       | 80.1%     | 500   | full    | 4.02 |        |
| resnet50 + model1:pp + model2:ppp1 + triplet | 256x256(224x224) | 50x1       | 80.6%     | 55    | full    | 4.02 |        |
| resnet50 + model1:pp + model2:ppp1 + triplet | 448x448          | 12x1       | 82.0%     | 130   | full    | 4.03 |        |

- experiments result on part dataset

| model                                          | image_size       | batch_size | accuracy | epoch | dataset | date | id  |
| ---------------------------------------------- | ---------------- | ---------- | -------- | ----- | ------- | ---- | --- |
| resnet50                                       | 256x256(224x224) | 80         | 88.8%    | 50    | part    | 3.19 |     |
| resnet50 + model1:cosine_similarity            | 256x256(224x224) | 40         | 90%      | 30    | part    | 3.21 |     |
| resnet50 + model1:inter+intra_dist             | 256x256(224x224) | 20         | 90.2%    | 100   | part    | 3.29 |     |
| resnet50 + model1:inter+intra_dist+tripletLoss | 256x256(224x224) | 13x3       | 91.6%    | 50    | pasd rt | 3.30 |     |

- experiments on C20

| model                                                                   | image_size       | batch_size | accuracy  | epoch | dataset | date | id  |
| ----------------------------------------------------------------------- | ---------------- | ---------- | --------- | ----- | ------- | ---- | --- |
| resnet50                                                                | 256x256(224x224) | 80         | 91.5%     | 40    | C20     | 3.31 |     |
| API-NET                                                                 | 256x256(224x224) | 13x3       | 86.8%     | 50    | C20     | 3.31 |     |
| API-NET                                                                 | 256x256(224x224) | 18x3       | 89.5%     | 50    | C20     | 3.31 |     |
| resnet50 + intra:raw_feature                                            | 256x256(224x224) | 13x3       | 90.7%     | 40    | C20     | 3.31 |     |
| resnet50 + intra:pp(2048-512-2048)                                      | 256x256(224x224) | 13x3       | 92.4%     | 40    | C20     | 3.31 |     |
| resnet50 + intra:pp(2048-512)                                           | 256x256(224x224) | 13x3       | 92.4%     | 40    | C20     | 3.31 |     |
| resnet50 + intra:pp(2048-512) drop=0.5                                  | 256x256(224x224) | 13x3       | 91.8%     | 40    | C20     | 3.31 |     |
| resnet50 + intra:pp(2048-512) inter:p0p0p1 triplet(p=2)                 | 256x256(224x224) | 13x3       | **93.2%** | 40    | C20     | 3.31 |     |
| resnet50 + intra:pp(2048-512) inter:p0p0p1 triplet(p=1)                 | 256x256(224x224) | 13x3       | 92.0%     | 40    | C20     | 3.31 |     |
| resnet50 + intra:rr' inter:raw_feature                                  | 256x256(224x224) | 13x3       | 83.3%     | 60    | C20     | 3.31 |     |
| resnet50 + intra:pp'(2048-512) inter:rrr triplet                        | 256x256(224x224) | 13x3       | 85.0%     | 60    | C20     | 3.31 |     |
| resnet50 + intra:pp'(2048-512) inter:p0p0p1 triplet(m=10)               | 256x256(224x224) | 13x3       | 91.7%     | 60    | C20     | 4.01 |     |
| resnet50 + intra:pp'(2048-512) inter:p0p0p1 triplet(m=20)               | 256x256(224x224) | 13x3       | 92.6%     | 60    | C20     | 4.01 |     |
| resnet50 + intra:pp'(2048-512) inter:p0p0p1 triplet(m=1)                | 256x256(224x224) | 12x1       | 83.9%     | 50    | C20     | 4.03 |     |
| resnet50 + intra:pp'(2048-512) inter:p0p0p1 triplet(m=1)                | 256x256(224x224) | 20x1       | 91.1%     | 50    | C20     | 4.03 |     |
| resnet50 + intra:pp'(2048-512) inter:p0p0p1 triplet(m=1)                | 256x256(224x224) | 20x2       | 91.8%     | 50    | C20     | 4.03 |     |
| resnet50 + intra:pp'(2048-512) inter:p0p0p1 triplet(m=2)                | 256x256(224x224) | 12x1       | 87.8%     | 50    | C20     | 4.03 |     |
| resnet50 + intra:pp'(2048-512) inter:p0p0p1 triplet(m=2)                | 256x256(224x224) | 20x1       | 88.8%     | 50    | C20     | 4.03 |     |
| resnet50 + intra:pp'(2048-512) inter:p0p0p1 triplet(m=2)                | 256x256(224x224) | 20x2       | 92.4%     | 50    | C20     | 4.03 |     |
| resnet50 + intra:pp'(2048-512) inter:p0p0p1 triplet(m=4)                | 256x256(224x224) | 12x1       | 84.7%     | 50    | C20     | 4.03 |     |
| resnet50 + intra:pp'(2048-512) inter:p0p0p1 triplet(m=4)                | 256x256(224x224) | 20x1       | 89.1%     | 50    | C20     | 4.03 |     |
| resnet50 + intra:pp'(2048-512) inter:p0p0p1 triplet(m=4)                | 256x256(224x224) | 10x2       | 88.2%     | 50    | C20     | 4.03 |     |
| resnet50 + intra:pp'(2048-512) inter:p0p0p1 triplet(m=4)                | 256x256(224x224) | 20x2       | 92.2%     | 50    | C20     | 4.03 |     |
| resnet50 + intra:pp'(2048-512) inter:p0p0p1 triplet(m=4)                | 256x256(224x224) | 15x3       | 92.0%     | 50    | C20     | 4.03 |     |
| resnet50 + intra:pp'(2048-512) inter:p0p0p1 triplet(m=4)                | 256x256(224x224) | 18x3       | 92.0%     | 50    | C20     | 4.03 |     |
| resnet50 + intra:pp'(2048-512) inter:p0p0p1 triplet(m=8)                | 256x256(224x224) | 12x1       | -         | 50    | C20     | 4.03 |     |
| resnet50 + intra:pp'(2048-512) inter:p0p0p1 triplet(m=8)                | 256x256(224x224) | 20x1       | 88.3%     | 50    | C20     | 4.05 |     |
| resnet50 + intra:pp'(2048-512) inter:p0p0p1 triplet(m=8)                | 256x256(224x224) | 20x2       | 91.5%     | 50    | C20     | 4.03 |     |
| resnet50 + intra:pp'(2048-512) inter:p0p0p1 triplet(m=16)               | 256x256(224x224) | 12x1       | -         | 50    | C20     | 4.03 |     |
| resnet50 + intra:pp'(2048-512) inter:p0p0p1 triplet(m=16)               | 256x256(224x224) | 20x1       | -         | 50    | C20     | 4.03 |     |
| resnet50 + intra:pp'(2048-512) inter:p0p0p1 triplet(m=16)               | 256x256(224x224) | 20x2       | 91.7%     | 50    | C20     | 4.03 |     |
| resnet50 + intra:pp'(2048-512)                                          | 256x256(224x224) | 20x2       | 91.7%     | 50    | C20     | 4.03 |     |
| resnet50 + intra:pp'(2048-512) inter:p0p0p1 triplet(m=32)               | 256x256(224x224) | 12x1       | -         | 50    | C20     | 4.03 |     |
| resnet50 + intra:pp'(2048-512) inter:p0p0p1 triplet(m=32)               | 256x256(224x224) | 20x1       | -         | 50    | C20     | 4.03 |     |
| resnet50 + intra:pp'(2048-512) inter:p0p0p1 triplet(m=32)               | 256x256(224x224) | 20x2       | -         | 50    | C20     | 4.03 |     |
| resnet50 + intra:pp'(2048-512) inter:p0p0p1 triplet(m=64)               | 256x256(224x224) | 12x1       | -         | 50    | C20     | 4.03 |     |
| resnet50 + intra:pp'(2048-512) inter:triplet(p0:512 p1:512)             | 256x256(224x224) | 13x3       | 91.8%     | 50    | C20     | 4.06 |     |
| intra:pp'(2048-512) inter:triplet(p0:512 p1:512) sample circle-repeat   | 256x256(224x224) | 52         | 92.2%     | 50    | C20     | 4.06 |     |
| resnet50 + intra:pp'(2048-512) inter:maximum(raw) triplet()             | 256x256(224x224) | 20x2       | 91.3%     | 50    | C20     | 4.06 |     |
| intra:pp'(2048-512) inter:lerp(raw, 0.5) triplet()                      | 256x256(224x224) | 20x2       | 91.3%     | 50    | C20     | 4.06 |     |
| intra:pp'(2048-512) inter:lerp(raw, 0.5) triplet() sample circle-repeat | 256x256(224x224) | 20x2       | 92.0%     | 50    | C20     | 4.06 |     |
| intra:pp'(2048-512) inter:lerp(raw, 0.5) triplet() sample random-class  | 256x256(224x224) | 8x5        | 92.0%     | 50    | C20     | 4.06 |     |
| intra:pp'(2048-512) inter:lerp(raw, 0.5) triplet() sample circle-repeat | 256x256(224x224) | 52         | 92.6%     | 50    | C20     | 4.06 |     |


# ????????????

1. ??????????????????
   1. ???????????????????????? ok
   2. ???Acc?????? ok
   3. BatchSampler????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
2. Pytorch??????????????????
   1. ????????????-?????????????????????
   2. ?????????????????????