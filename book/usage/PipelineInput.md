# Pipeline Input

The pipeline takes as input an experiment directory and a configuration file. The experiment directory should contain raw images stored in a single subdirectory (e.g. 'raw'). If this raw directory itself contains subdirectories, the pipeline will process them independently and mirror this structure in its output. Information on how to build the configuration file can be found in the [Running your first pipeline](https://spsalmon.github.io/align_pipeline/getting-started/runningfirstpipeline/) section.

The raw images should be in OME-TIFF format. Most importantly, all images for a given position at a given time should be stored in the same N-dimension OME-TIFF file. All file names should contain a unique identifier for the position (e.g. Point1, Point2, etc.) and for the time (e.g. Time1, Time2, etc.). The pipeline will use regular expressions to extract those indices from the file names, so you can adapt the naming scheme to your needs as long as you provide the right regular expressions in the configuration file.

If your experiment contains different imaging modalities (e.g. a Z-stack every hour and a picture of each worm every 10 minutes), you should split them in different raw directories (e.g. raw and raw_stack). You can then run a pipeline for each of those directories and merge them at the end simply by joining the two dataframes.
