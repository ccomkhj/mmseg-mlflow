## Introduction

MMSegmentation is an open source semantic segmentation toolbox based on PyTorch.
It is a part of the [OpenMMLab](https://openmmlab.com/) project.

The master branch works with **PyTorch 1.5+**.

## My Contribution

MMSegmentation doesn't support hyperOpt (hyper parameter tuner) natively.
`hyper_opt.py` explains how to run hyper parameter tuning.
`tools/train_arg_in` gives an example how to modify the source code.

if you want to integrate to MLflow, then follow the shell script below.

```bash
export AWS_ACCESS_KEY_ID= 
export AWS_SECRET_ACCESS_KEY= 
export AWS_DEFAULT_REGION= 
export MLFLOW_TRACKING_URI= # if you have a remote server. If you run server locally, then make it empty.
export MLFLOW_S3_ENDPOINT_URL= "http://s3.eu-central-1.amazonaws.com"  # This is the case of Frankfurt datacenter. 
python hyper_opt.py
```

In case, mlflowhook is also applied to track, then edit at `mmcv/mmcv/runner/hooks/logger/mlflow.py
```python
@master_only
    def after_run(self, runner) -> None:
        if self.log_model:
            self.mlflow_pytorch.log_model(
                runner.model,
                'models',
                pip_requirements=[f'torch=={TORCH_VERSION}'])
        self.mlflow.end_run() # add this line to conclude per loop.
```
