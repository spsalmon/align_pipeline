# align_pipeline

align_pipeline is a modular pipelining tool made to streamline the analysis
of *C. elegans* time lapse microscopy experiments. It includes:

- a robust and human readable configuration system for reproducible image
  analysis pipelines
- state of the art deep learning segmentation
- tools to easily retrain and fine tune said segmentation models
- a GUI for annotating time series (i.e. molts)
- convenient plotting in jupyter notebooks with ready made functions

It runs on a computing cluster or on your own machine, from the same
configuration file.

Head to the **Getting Started** section to learn how to [install the pipeline](https://spsalmon.github.io/align_pipeline/getting-started/installation/), [update the pipeline](https://spsalmon.github.io/align_pipeline/getting-started/update/), and how to [run your first pipeline](https://spsalmon.github.io/align_pipeline/getting-started/runningfirstpipeline/).

The **Usage** section covers the [configuration options](https://spsalmon.github.io/align_pipeline/usage/configuration/), [what a run produces](https://spsalmon.github.io/align_pipeline/usage/pipelineoutput/), [how to follow a run](https://spsalmon.github.io/align_pipeline/usage/monitoringruns/) and [how to use the GUI](https://spsalmon.github.io/align_pipeline/usage/usinggui/).

The **Building Blocks** section contains a [detailed explanation of all the parameters of every block](https://spsalmon.github.io/align_pipeline/building-blocks/buildingblock/), and the **Training** section explains how to [retrain the models](https://spsalmon.github.io/align_pipeline/training/training/).
