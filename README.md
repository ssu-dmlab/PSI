#### An example to run a 3-layer LightGCN

run LightGCN on **gowalla**, **amazon-book** dataset:

- command

`cd code`

`python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="gowalla" --topks="[20]" --recdim=64`

`python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="amazon-book" --topks="[20]" --recdim=64`


- log output

```shell
...
======================
EPOCH[5/1000]
BPR[sample time][16.2=15.84+0.42]
[saved][[BPR[aver loss1.128e-01]]
[0;30;43m[TEST][0m
{'precision': array([0.03315359]), 'recall': array([0.10711388]), 'ndcg': array([0.08940792])}
[TOTAL TIME] 35.9975962638855
...
======================
EPOCH[116/1000]
BPR[sample time][16.9=16.60+0.45]
[saved][[BPR[aver loss2.056e-02]]
[TOTAL TIME] 30.99874997138977
...
```

_NOTE_:

1. Even though we offer the code to split user-item matrix for matrix multiplication, we strongly suggest you don't enable it since it will extremely slow down the training speed.
2. If you feel the test process is slow, try to increase the ` testbatch` and enable `multicore`(Windows system may encounter problems with `multicore` option enabled)
3. Use `tensorboard` option, it's good.
4. Since we fix the seed(`--seed=2020` ) of `numpy` and `torch` in the beginning, if you run the command as we do above, you should have the exact output log despite the running time (check your output of _epoch 5_ and _epoch 116_).
