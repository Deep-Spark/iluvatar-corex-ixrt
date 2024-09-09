# IxRT open source tools

| Directory                                          | Description                                                  |
| -------------------------------------------------- | ------------------------------------------------------------ |
| [convert_old_ir_to_onnx](./convert_old_ir_to_onnx) | Tool for converting graph json + weights back to json. If you only have json+weights files, you can use this tool to help you convert them to onnx and then use new IxRT API |
| [optimizer](./optimizer) | Optimizer tool is a graph fusion tool integrated in ixrt, used to fuse ops in the ONNX graph into corresponding IxRT plugins, typically used in conjunction with IxRT. |
