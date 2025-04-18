# Frenet-Serret Frame-based Decomposition for Part Segmentation of 3D Curvilinear Structures
## Please find the released dataset at https://huggingface.co/datasets/pytc/DenSpineEM.


# Experiments on DenSpineEM Benchmark

## Average Results:  Mean ± Standard Deviation (95% Confidence Interval)
| Model | Dataset | Spine IoU | Trunk IoU | Spine Dice | Trunk Dice | Spine Accuracy | Average Spine Recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PointNet++ | M50 | 0.6951 ± 0.0314 (0.6676 - 0.7226) | 0.9484 ± 0.0223 (0.9289 - 0.9680) | 0.8004 ± 0.0286 (0.7753 - 0.8254) | 0.9724 ± 0.0132 (0.9608 - 0.9840) | 0.8703 ± 0.0189 (0.8537 - 0.8869) | 0.8644 ± 0.0150 (0.8513 - 0.8775) | 
| PointNet++ | M10 | 0.7329 ± 0.0217 (0.7139 - 0.7518) | 0.8948 ± 0.0095 (0.8865 - 0.9031) | 0.8428 ± 0.0147 (0.8299 - 0.8557) | 0.9433 ± 0.0055 (0.9384 - 0.9481) | 0.7866 ± 0.0318 (0.7587 - 0.8145) | 0.8058 ± 0.0265 (0.7826 - 0.8291) | 
| PointNet++ | H10 | 0.6056 ± 0.0285 (0.5807 - 0.6306) | 0.8985 ± 0.0120 (0.8880 - 0.9090) | 0.7419 ± 0.0247 (0.7203 - 0.7636) | 0.9454 ± 0.0067 (0.9395 - 0.9513) | 0.6673 ± 0.0536 (0.6203 - 0.7143) | 0.7417 ± 0.0329 (0.7129 - 0.7705) | 
| PointNet++ w. FFD | M50 | 0.8260 ± 0.0409 (0.7902 - 0.8618) | 0.9808 ± 0.0034 (0.9778 - 0.9838) | 0.8928 ± 0.0283 (0.8681 - 0.9176) | 0.9903 ± 0.0018 (0.9887 - 0.9918) | 0.8992 ± 0.0212 (0.8806 - 0.9178) | 0.8861 ± 0.0155 (0.8725 - 0.8997) | 
| PointNet++ w. FFD | M10 | 0.8740 ± 0.0087 (0.8663 - 0.8816) | 0.9566 ± 0.0030 (0.9540 - 0.9592) | 0.9319 ± 0.0051 (0.9274 - 0.9365) | 0.9777 ± 0.0016 (0.9763 - 0.9791) | 0.8656 ± 0.0233 (0.8452 - 0.8860) | 0.8546 ± 0.0210 (0.8362 - 0.8729) | 
| PointNet++ w. FFD | H10 | 0.6829 ± 0.0208 (0.6647 - 0.7011) | 0.9286 ± 0.0071 (0.9224 - 0.9349) | 0.7927 ± 0.0202 (0.7751 - 0.8104) | 0.9626 ± 0.0040 (0.9591 - 0.9661) | 0.7850 ± 0.0240 (0.7640 - 0.8061) | 0.8095 ± 0.0218 (0.7904 - 0.8286) | 
| RandLA-Net | M50 | 0.1488 ± 0.0252 (0.1267 - 0.1710) | 0.6397 ± 0.1040 (0.5485 - 0.7309) | 0.2463 ± 0.0359 (0.2148 - 0.2778) | 0.7660 ± 0.0840 (0.6924 - 0.8396) | 0.4304 ± 0.1692 (0.2821 - 0.5788) | 0.4840 ± 0.1647 (0.3396 - 0.6283) | 
| RandLA-Net | M10 | 0.2433 ± 0.0332 (0.2142 - 0.2724) | 0.4821 ± 0.0694 (0.4213 - 0.5429) | 0.3794 ± 0.0419 (0.3427 - 0.4162) | 0.6401 ± 0.0668 (0.5815 - 0.6986) | 0.4885 ± 0.1266 (0.3776 - 0.5995) | 0.5396 ± 0.1219 (0.4328 - 0.6464) | 
| RandLA-Net | H10 | 0.2202 ± 0.0256 (0.1977 - 0.2426) | 0.5597 ± 0.0768 (0.4924 - 0.6271) | 0.3449 ± 0.0326 (0.3164 - 0.3735) | 0.7078 ± 0.0631 (0.6525 - 0.7631) | 0.4907 ± 0.1691 (0.3425 - 0.6390) | 0.5382 ± 0.1632 (0.3951 - 0.6813) | 
| RandLA-Net w. FFD | M50 | 0.3710 ± 0.1782 (0.2149 - 0.5272) | 0.8665 ± 0.0487 (0.8238 - 0.9091) | 0.4938 ± 0.1781 (0.3377 - 0.6500) | 0.9252 ± 0.0288 (0.9000 - 0.9505) | 0.4867 ± 0.2121 (0.3008 - 0.6725) | 0.5524 ± 0.1807 (0.3940 - 0.7108) | 
| RandLA-Net w. FFD | M10 | 0.4406 ± 0.1332 (0.3239 - 0.5574) | 0.7857 ± 0.0521 (0.7400 - 0.8314) | 0.5884 ± 0.1337 (0.4712 - 0.7056) | 0.8762 ± 0.0337 (0.8466 - 0.9057) | 0.3713 ± 0.1404 (0.2483 - 0.4944) | 0.4847 ± 0.1242 (0.3759 - 0.5936) | 
| RandLA-Net w. FFD | H10 | 0.3745 ± 0.1234 (0.2663 - 0.4826) | 0.7900 ± 0.0505 (0.7457 - 0.8343) | 0.5168 ± 0.1261 (0.4062 - 0.6273) | 0.8796 ± 0.0312 (0.8522 - 0.9070) | 0.4211 ± 0.1916 (0.2531 - 0.5890) | 0.5299 ± 0.1626 (0.3874 - 0.6724) | 
| PointTransformer | M50 | 0.8807 ± 0.0242 (0.8595 - 0.9019) | 0.9829 ± 0.0027 (0.9804 - 0.9853) | 0.9261 ± 0.0204 (0.9082 - 0.9439) | 0.9912 ± 0.0015 (0.9899 - 0.9925) | 0.9594 ± 0.0085 (0.9519 - 0.9668) | 0.9464 ± 0.0087 (0.9388 - 0.9540) | 
| PointTransformer | M10 | 0.8377 ± 0.0132 (0.8262 - 0.8493) | 0.9337 ± 0.0052 (0.9292 - 0.9383) | 0.9080 ± 0.0078 (0.9011 - 0.9149) | 0.9650 ± 0.0028 (0.9625 - 0.9675) | 0.9121 ± 0.0106 (0.9028 - 0.9214) | 0.9020 ± 0.0088 (0.8943 - 0.9097) | 
| PointTransformer | H10 | 0.7377 ± 0.0153 (0.7243 - 0.7511) | 0.9282 ± 0.0051 (0.9237 - 0.9327) | 0.8395 ± 0.0111 (0.8298 - 0.8493) | 0.9618 ± 0.0030 (0.9591 - 0.9644) | 0.8255 ± 0.0132 (0.8140 - 0.8371) | 0.8478 ± 0.0104 (0.8387 - 0.8568) | 
| PointTransformer w. FFD | M50 | 0.9074 ± 0.0250 (0.8855 - 0.9293) | 0.9900 ± 0.0029 (0.9875 - 0.9925) | 0.9443 ± 0.0211 (0.9258 - 0.9628) | 0.9949 ± 0.0015 (0.9936 - 0.9962) | 0.9545 ± 0.0138 (0.9424 - 0.9666) | 0.9382 ± 0.0155 (0.9246 - 0.9518) | 
| PointTransformer w. FFD | M10 | 0.9172 ± 0.0022 (0.9153 - 0.9192) | 0.9721 ± 0.0008 (0.9713 - 0.9728) | 0.9561 ± 0.0013 (0.9550 - 0.9572) | 0.9858 ± 0.0004 (0.9854 - 0.9861) | 0.9586 ± 0.0035 (0.9555 - 0.9617) | 0.9303 ± 0.0039 (0.9269 - 0.9337) | 
| PointTransformer w. FFD | H10 | 0.7757 ± 0.0130 (0.7642 - 0.7871) | 0.9575 ± 0.0019 (0.9559 - 0.9592) | 0.8663 ± 0.0098 (0.8578 - 0.8749) | 0.9782 ± 0.0010 (0.9774 - 0.9791) | 0.8368 ± 0.0148 (0.8238 - 0.8498) | 0.8482 ± 0.0118 (0.8379 - 0.8586) | 

## Fold 0
| Model | Dataset | Spine IoU | Trunk IoU | Spine Dice | Trunk Dice | Spine Accuracy | Average Spine Recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PointNet++ | M50 | 0.6685 | 0.9071 | 0.7897 | 0.9475 | 0.8877 | 0.8866 | 
| PointNet++ | M10 | 0.7384 | 0.9006 | 0.8464 | 0.9468 | 0.8153 | 0.8263 | 
| PointNet++ | H10 | 0.5515 | 0.8747 | 0.6960 | 0.9321 | 0.7508 | 0.7990 | 
| PointNet++ w. FFD | M50 | 0.8166 | 0.9760 | 0.8768 | 0.9878 | 0.9130 | 0.9022 | 
| PointNet++ w. FFD | M10 | 0.8785 | 0.9596 | 0.9343 | 0.9793 | 0.9013 | 0.8846 | 
| PointNet++ w. FFD | H10 | 0.6738 | 0.9236 | 0.7817 | 0.9599 | 0.8255 | 0.8451 | 
| RandLA-Net | M50 | 0.1914 | 0.4588 | 0.3042 | 0.6193 | 0.7089 | 0.7582 | 
| RandLA-Net | M10 | 0.2877 | 0.3538 | 0.4289 | 0.5128 | 0.7134 | 0.7580 | 
| RandLA-Net | H10 | 0.2618 | 0.4404 | 0.3986 | 0.6052 | 0.7668 | 0.8047 | 
| RandLA-Net w. FFD | M50 | 0.6432 | 0.9409 | 0.7565 | 0.9686 | 0.6772 | 0.7015 | 
| RandLA-Net w. FFD | M10 | 0.6174 | 0.8491 | 0.7535 | 0.9144 | 0.4586 | 0.5505 | 
| RandLA-Net w. FFD | H10 | 0.5934 | 0.8881 | 0.7398 | 0.9396 | 0.5974 | 0.6682 | 
| PointTransformer | M50 | 0.8709 | 0.9794 | 0.9169 | 0.9894 | 0.9541 | 0.9412 | 
| PointTransformer | M10 | 0.8469 | 0.9372 | 0.9141 | 0.9670 | 0.9299 | 0.9173 | 
| PointTransformer | H10 | 0.7368 | 0.9273 | 0.8372 | 0.9613 | 0.8442 | 0.8644 | 
| PointTransformer w. FFD | M50 | 0.8712 | 0.9852 | 0.9083 | 0.9925 | 0.9525 | 0.9312 | 
| PointTransformer w. FFD | M10 | 0.9152 | 0.9711 | 0.9550 | 0.9852 | 0.9554 | 0.9256 | 
| PointTransformer w. FFD | H10 | 0.7820 | 0.9586 | 0.8718 | 0.9788 | 0.8380 | 0.8417 | 

## Fold 1
| Model | Dataset | Spine IoU | Trunk IoU | Spine Dice | Trunk Dice | Spine Accuracy | Average Spine Recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PointNet++ | M50 | 0.6829 | 0.9449 | 0.7963 | 0.9713 | 0.8758 | 0.8729 | 
| PointNet++ | M10 | 0.7120 | 0.8790 | 0.8294 | 0.9340 | 0.8280 | 0.8430 | 
| PointNet++ | H10 | 0.6188 | 0.9069 | 0.7499 | 0.9502 | 0.6636 | 0.7276 | 
| PointNet++ w. FFD | M50 | 0.8179 | 0.9812 | 0.8849 | 0.9905 | 0.9088 | 0.8938 | 
| PointNet++ w. FFD | M10 | 0.8752 | 0.9571 | 0.9328 | 0.9780 | 0.8758 | 0.8599 | 
| PointNet++ w. FFD | H10 | 0.6801 | 0.9276 | 0.7892 | 0.9621 | 0.7882 | 0.8116 | 
| RandLA-Net | M50 | 0.1607 | 0.6032 | 0.2677 | 0.7350 | 0.4462 | 0.5090 | 
| RandLA-Net | M10 | 0.2561 | 0.5579 | 0.4019 | 0.7047 | 0.4650 | 0.5203 | 
| RandLA-Net | H10 | 0.2108 | 0.5573 | 0.3314 | 0.7094 | 0.4760 | 0.5158 | 
| RandLA-Net w. FFD | M50 | 0.1880 | 0.7935 | 0.3046 | 0.8827 | 0.2297 | 0.3421 | 
| RandLA-Net w. FFD | M10 | 0.2446 | 0.6991 | 0.3809 | 0.8183 | 0.2006 | 0.3192 | 
| RandLA-Net w. FFD | H10 | 0.2377 | 0.7547 | 0.3746 | 0.8558 | 0.2236 | 0.3420 | 
| PointTransformer | M50 | 0.9173 | 0.9861 | 0.9560 | 0.9930 | 0.9714 | 0.9583 | 
| PointTransformer | M10 | 0.8410 | 0.9349 | 0.9091 | 0.9655 | 0.9140 | 0.9032 | 
| PointTransformer | H10 | 0.7433 | 0.9294 | 0.8444 | 0.9625 | 0.8162 | 0.8406 | 
| PointTransformer w. FFD | M50 | 0.9013 | 0.9916 | 0.9434 | 0.9958 | 0.9637 | 0.9484 | 
| PointTransformer w. FFD | M10 | 0.9144 | 0.9714 | 0.9544 | 0.9854 | 0.9650 | 0.9327 | 
| PointTransformer w. FFD | H10 | 0.7630 | 0.9545 | 0.8580 | 0.9766 | 0.8505 | 0.8637 | 

## Fold 2
| Model | Dataset | Spine IoU | Trunk IoU | Spine Dice | Trunk Dice | Spine Accuracy | Average Spine Recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PointNet++ | M50 | 0.6595 | 0.9706 | 0.7543 | 0.9850 | 0.8474 | 0.8455 | 
| PointNet++ | M10 | 0.7724 | 0.9076 | 0.8697 | 0.9504 | 0.7866 | 0.8058 | 
| PointNet++ | H10 | 0.6048 | 0.9022 | 0.7400 | 0.9477 | 0.6978 | 0.7558 | 
| PointNet++ w. FFD | M50 | 0.7597 | 0.9799 | 0.8536 | 0.9898 | 0.8571 | 0.8567 | 
| PointNet++ w. FFD | M10 | 0.8862 | 0.9596 | 0.9393 | 0.9793 | 0.8694 | 0.8655 | 
| PointNet++ w. FFD | H10 | 0.6605 | 0.9201 | 0.7709 | 0.9578 | 0.7788 | 0.8164 | 
| RandLA-Net | M50 | 0.1382 | 0.7089 | 0.2290 | 0.8201 | 0.4675 | 0.4947 | 
| RandLA-Net | M10 | 0.2576 | 0.4986 | 0.3989 | 0.6585 | 0.4968 | 0.5383 | 
| RandLA-Net | H10 | 0.2342 | 0.5452 | 0.3615 | 0.6999 | 0.5272 | 0.5684 | 
| RandLA-Net w. FFD | M50 | 0.2062 | 0.8690 | 0.3166 | 0.9284 | 0.3182 | 0.3916 | 
| RandLA-Net w. FFD | M10 | 0.3388 | 0.7604 | 0.4948 | 0.8619 | 0.2611 | 0.3876 | 
| RandLA-Net w. FFD | H10 | 0.2941 | 0.7754 | 0.4350 | 0.8720 | 0.2556 | 0.3944 | 
| PointTransformer | M50 | 0.8701 | 0.9854 | 0.9154 | 0.9926 | 0.9535 | 0.9334 | 
| PointTransformer | M10 | 0.8531 | 0.9400 | 0.9170 | 0.9684 | 0.9140 | 0.9028 | 
| PointTransformer | H10 | 0.7494 | 0.9349 | 0.8469 | 0.9656 | 0.8349 | 0.8554 | 
| PointTransformer w. FFD | M50 | 0.8947 | 0.9892 | 0.9396 | 0.9945 | 0.9297 | 0.9108 | 
| PointTransformer w. FFD | M10 | 0.9193 | 0.9723 | 0.9574 | 0.9859 | 0.9554 | 0.9271 | 
| PointTransformer w. FFD | H10 | 0.7737 | 0.9569 | 0.8648 | 0.9779 | 0.8100 | 0.8311 | 

## Fold 3
| Model | Dataset | Spine IoU | Trunk IoU | Spine Dice | Trunk Dice | Spine Accuracy | Average Spine Recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PointNet++ | M50 | 0.7303 | 0.9573 | 0.8314 | 0.9778 | 0.8487 | 0.8504 | 
| PointNet++ | M10 | 0.7173 | 0.8934 | 0.8312 | 0.9426 | 0.7580 | 0.7828 | 
| PointNet++ | H10 | 0.6325 | 0.9053 | 0.7680 | 0.9492 | 0.5981 | 0.7065 | 
| PointNet++ w. FFD | M50 | 0.8799 | 0.9867 | 0.9325 | 0.9933 | 0.9117 | 0.8884 | 
| PointNet++ w. FFD | M10 | 0.8696 | 0.9551 | 0.9295 | 0.9769 | 0.8344 | 0.8377 | 
| PointNet++ w. FFD | H10 | 0.6782 | 0.9311 | 0.7913 | 0.9641 | 0.7819 | 0.7926 | 
| RandLA-Net | M50 | 0.1180 | 0.7601 | 0.2007 | 0.8608 | 0.1972 | 0.2528 | 
| RandLA-Net | M10 | 0.1900 | 0.5217 | 0.3086 | 0.6794 | 0.3248 | 0.3832 | 
| RandLA-Net | H10 | 0.1875 | 0.6816 | 0.3032 | 0.8031 | 0.2396 | 0.2934 | 
| RandLA-Net w. FFD | M50 | 0.3062 | 0.8426 | 0.4533 | 0.9083 | 0.4217 | 0.5114 | 
| RandLA-Net w. FFD | M10 | 0.4758 | 0.8214 | 0.6332 | 0.9003 | 0.3439 | 0.4921 | 
| RandLA-Net w. FFD | H10 | 0.3344 | 0.7818 | 0.4794 | 0.8764 | 0.3291 | 0.4770 | 
| PointTransformer | M50 | 0.8976 | 0.9801 | 0.9426 | 0.9897 | 0.9503 | 0.9467 | 
| PointTransformer | M10 | 0.8151 | 0.9249 | 0.8945 | 0.9602 | 0.9013 | 0.8928 | 
| PointTransformer | H10 | 0.7087 | 0.9192 | 0.8190 | 0.9564 | 0.8255 | 0.8406 | 
| PointTransformer w. FFD | M50 | 0.9421 | 0.9938 | 0.9696 | 0.9969 | 0.9563 | 0.9477 | 
| PointTransformer w. FFD | M10 | 0.9200 | 0.9734 | 0.9577 | 0.9865 | 0.9586 | 0.9363 | 
| PointTransformer w. FFD | H10 | 0.7973 | 0.9602 | 0.8820 | 0.9796 | 0.8505 | 0.8590 | 

## Fold 4
| Model | Dataset | Spine IoU | Trunk IoU | Spine Dice | Trunk Dice | Spine Accuracy | Average Spine Recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PointNet++ | M50 | 0.7345 | 0.9622 | 0.8301 | 0.9806 | 0.8919 | 0.8665 | 
| PointNet++ | M10 | 0.7242 | 0.8933 | 0.8373 | 0.9425 | 0.7452 | 0.7714 | 
| PointNet++ | H10 | 0.6206 | 0.9032 | 0.7558 | 0.9477 | 0.6262 | 0.7195 | 
| PointNet++ w. FFD | M50 | 0.8560 | 0.9804 | 0.9164 | 0.9900 | 0.9054 | 0.8894 | 
| PointNet++ w. FFD | M10 | 0.8603 | 0.9517 | 0.9238 | 0.9750 | 0.8471 | 0.8251 | 
| PointNet++ w. FFD | H10 | 0.7221 | 0.9409 | 0.8304 | 0.9693 | 0.7508 | 0.7818 | 
| RandLA-Net | M50 | 0.1360 | 0.6677 | 0.2299 | 0.7951 | 0.3324 | 0.4052 | 
| RandLA-Net | M10 | 0.2251 | 0.4786 | 0.3589 | 0.6448 | 0.4427 | 0.4982 | 
| RandLA-Net | H10 | 0.2065 | 0.5741 | 0.3300 | 0.7214 | 0.4441 | 0.5087 | 
| RandLA-Net w. FFD | M50 | 0.5117 | 0.8864 | 0.6381 | 0.9381 | 0.7865 | 0.8156 | 
| RandLA-Net w. FFD | M10 | 0.5266 | 0.7987 | 0.6796 | 0.8858 | 0.5924 | 0.6742 | 
| RandLA-Net w. FFD | H10 | 0.4126 | 0.7500 | 0.5551 | 0.8543 | 0.6997 | 0.7677 | 
| PointTransformer | M50 | 0.8477 | 0.9833 | 0.8994 | 0.9915 | 0.9676 | 0.9525 | 
| PointTransformer | M10 | 0.8327 | 0.9317 | 0.9052 | 0.9640 | 0.9013 | 0.8938 | 
| PointTransformer | H10 | 0.7504 | 0.9301 | 0.8501 | 0.9629 | 0.8069 | 0.8378 | 
| PointTransformer w. FFD | M50 | 0.9276 | 0.9900 | 0.9606 | 0.9949 | 0.9703 | 0.9528 | 
| PointTransformer w. FFD | M10 | 0.9171 | 0.9721 | 0.9560 | 0.9858 | 0.9586 | 0.9297 | 
| PointTransformer w. FFD | H10 | 0.7623 | 0.9574 | 0.8549 | 0.9782 | 0.8349 | 0.8457 | 

