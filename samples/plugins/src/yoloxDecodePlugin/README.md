# YoloV3Decoder

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Algorithms](#algorithms)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description
The `YoloXDecoderPlugin` implements YoloX decoder proposed in [paper](https://arxiv.org/abs/2107.08430). Specifically:
- format: NCHW for both input and output
- dtype: FLOAT16 input and FLOAT32 output

### Structure

The `YoloXDecoderPlugin` takes three inputs, boxes input, box confidence input and class confidence input.

- `Boxes input`
The boxes input are of shape `[batch_size, number_box_parameters, H, W]`.

- `Boxes Confidence input`
The boxes confidence input are of shape `[batch_size, 1, H, W]`.

- `Classes Confidence input`
The classes confidence input are of shape `[batch_size, number_classes, H, W]`.


With the input, the plugin generates 2 output:

- `Boxes output`
The boxes output are of shape `[batch_size, number_box_parameters, 1, number_boxes]`.

- `Scores output`
The scores output are of shape `[batch_size, number_classes, number_boxes]`.


## Parameters
The plugin has the plugin creator class YoloXDecoderPluginCreator and the plugin class YoloXDecoderPlugin.

The YoloXDecoderPlugin plugin instance is created using the following parameters:

| Type     | Parameter                | Description
|----------|--------------------------|--------------------------------------------------------
|`int32_t` |`num_class`               | number of classes of yolox classification
|`int32_t` |`stride`                  | the ratio by which it downsamples the input

## Algorithms

TODO

## Additional resources

The following resources provide a deeper understanding of the `YoloXDecoderPlugin` plugin:

- [YoloX paper](https://arxiv.org/abs/2107.08430)

## License

TODO

## Changelog

None

## Known issues

None
