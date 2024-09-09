# Converter tool to generate standard loadable onnx

## Background
In beginning versions of IxRT, we used graph json and its weight file to represent neural network model. However, with the progress of IxRT's development, this intermediate representation is no longer supported. Hence we provided a convert script for you to migrate original graph json and weight files to an onnx file. And, we recommend you to use IxRT new API to write inference code.


## Usage
Make sure you have `onnx` installed
```bash
pip3 install onnx
```

Then, just run this single script
```bash
python3 convert.py \
            --json_path path/to/graph/json \
            --weight_path path/to/weight \
            --onnx output/onnx/name \
            --input_info input_name1:dtype1:1x2x3,input_name2:dtype2:12x13 --output_info output_name1:dtype1
```

Parameter explained:
- `--json_path`: path to your graph json file
- `--weight_path`: path to your weight file
- `--onnx`: path to save your output onnx file

There may be no "input"/"output" in your previous graph json file, so please use these two arguments to specify model input/output
- `--input_info`: specify model i/o info, format should be `name1:dtype1:shape1,name2:dtype2:shape2`, use `,` to seperate different inputs, use `:` to seperate different attributes. For inputs, shape is a must
- `--output_info`: specify model i/o info, format should be `name1:dtype1:shape1,name2:dtype2:shape2`, use `,` to seperate different outputs, use `:` to seperate different attributes. For outputs, shape is not a must
