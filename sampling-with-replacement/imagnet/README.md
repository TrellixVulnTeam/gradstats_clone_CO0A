Training Resnet-50 with imagenet data in 90 epochs <br>
batch size 2048, learning rate 0.8, 8 gpus.

Results table:

| Epoch | Without Replacement | With Replacement |
| --- | --- | --- |
| 0 | Acc@1 1.250 Acc@5 4.664 | Acc@1 1.212 Acc@5 4.698 |
| 5 | Acc@1 29.734 Acc@5 54.930 | Acc@1 32.538 Acc@5 58.160 |
| 10 | Acc@1 42.184 Acc@5 69.542 | Acc@1 42.710 Acc@5 69.728 |
| 30 | Acc@1 66.424 Acc@5 87.522 | Acc@1 66.134 Acc@5 87.344 |
| 60 | Acc@1 73.162 Acc@5 91.448 | Acc@1 73.300 Acc@5 91.458 |
| 89 | Acc@1 73.884 Acc@5 91.770 | Acc@1 73.788 Acc@5 91.692 |
| highest | (87)Acc@1 74.554 Acc@5 92.042 | (77)Acc@1 74.470 Acc@5 92.062 |
