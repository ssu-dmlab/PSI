## Improving negative sampling on recommendation

### Experiments settings
- 실험은 script 폴더 안의 .sh 를 이용해서 수행.
    - 스크립트 예시
    ```bash
    #!/bin/bash
    #SBATCH --gpus=1
    #SBATCH --output=output/experiment_1

    python -m code.LFM --dataset ...
    ```
    - 실행법
    ```bash
    >> pwd
    /home/yourname/PSI
    >> sbatch script/your_experiment.sh
    ```
- 필요에 따라 yaml 등을 활용해 arguments 관리
- 실험결과는 output 디렉토리 아래에서 관리.
    - 유의미한 실험은 수행 후 노션에서 별도로 설명을 달아놓기

### 소스코드 내 파일 경로 지정법
- 매번 실행 위치에 따라 파일 import 경로가 달라지는 불편함을 방지하기 위해 프로젝트 디렉토리에서 모듈로 실행할 경우 기준을 다음과 같은 상대경로로 통일할 수 있다.
```python
dataset = 'amazon-book'
path = f'data/{dataset}/train.txt'
```

---
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
