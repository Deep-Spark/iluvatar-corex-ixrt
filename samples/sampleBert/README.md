## How to Use (Bert-Large-Squad as example)
### Prepare Environment
``` shell
bash script/prepare.sh v1_1
```

### build engine fp16
``` shell
# use --bs to set max_batch_size (dynamic)
bash script/build_engine --bs 32
```

### Test the performance of model fp16
``` shell
bash script/perf.sh --bs {batch_size}
```

### Test the accuracy of model fp16
``` shell
bash script/inference_squad.sh --bs {batch_size}
```

### build engine int8
``` shell
bash script/build_engine --bs 32 --int8
```

### Test the performance of model int8
``` shell
bash script/perf.sh --bs {batch_size} --int8
```

### Test the accuracy of model int8
``` shell
bash script/inference_squad.sh --bs {batch_size} --int8
```
