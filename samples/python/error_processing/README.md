# Error handling sample

## Description

You can inherit `IErrorRecorder` and register it to IxRT objects (builder, engine, context, network etc.) to retrieve the errors occured in IxRT.

This example intend to load a wrong engine file to get the error message of IxRT.


## How to run
```bash
python3 main.py
```

After running, you could see message like:
```
#0 error code: ErrorCodeTRT.UNSPECIFIED_ERROR
#0 error desc: [engine.cc:182 (IsEngineVerified) ] condition check failed at:h->magic == IXRT_MAGIC_NUMBER
```

That means error happens in IxRT, you may want to process the error with your logic.
