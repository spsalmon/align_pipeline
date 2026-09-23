# Plotting

We provide a number of plotting utilities for visualizing the results of the pipeline. Those functions are part of the `align_toolbox.plotting` module. Full documentation of those function is available [here](https://align-toolbox.readthedocs.io/en/latest/align_toolbox_plotting.html).

## Specifying experimental conditions

The plotting modules requires the user to map the positions contained in each experiment to an experimental condition. This is done by writing a YAML file that contains the mapping. This file is typically located in a directory called `doc` inside of the experiment directory. An example of such file is shown below:

```yaml
group_by: "point_range"
conditions:
  - strain:
      - wBT615
    point_range:
      - [0, 36]
    description:
      - eat-2
  - strain:
      - wBT125
    point_range:
      - [37, 76]
    description:
      - WT
```

More complex experiments can also we written in factorized form, as shown below:

```yaml
group_by: "point_range"
conditions:
  - auxin_concentration: 500
    strain:
      - wBT439
      - wBT186
      - wBT438
      - wBT437
    point_range:
      - [0, 23]
      - [24, 46]
      - [47, 69]
      - [70, 91]
    description:
      - col-10:tir, 500uM IAA
      - raga-1:aid, col-10:tir, 500uM IAA
      - yap-1:aid, col-10:tir, 500uM IAA
      - yap-1:aid, raga-1:aid, col-10:tir, 500uM IAA
  - auxin_concentration: 250
    strain:
      - wBT439
      - wBT186
      - wBT438
      - wBT437
    point_range:
      - [92, 111]
      - [112, 137]
      - [138, 162]
      - [163, 187]
    description:
      - col-10:tir, 250uM IAA
      - raga-1:aid, col-10:tir, 250uM IAA
      - yap-1:aid, col-10:tir, 250uM IAA
      - yap-1:aid, raga-1:aid, col-10:tir, 250uM IAA
  - auxin_concentration: 100
    strain:
      - wBT439
      - wBT186
      - wBT438
      - wBT437
    point_range:
      - [188, 211]
      - [212, 234]
      - [[235, 256], [281, 303]]
      - [[257, 280], [304, 333], [334, 359]]
    description:
      - col-10:tir, 100uM IAA
      - raga-1:aid, col-10:tir, 100uM IAA
      - yap-1:aid, col-10:tir, 100uM IAA
      - yap-1:aid, raga-1:aid, col-10:tir, 100uM IAA
  - auxin_concentration: 0
    strain:
      - wBT439
      - wBT186
      - wBT438
      - wBT437
    point_range:
      - [360, 382]
      - [383, 405]
      - [406, 429]
      - [430, 453]
    description:
      - col-10:tir, no IAA
      - WT yap-1, raga-1:aid, col-10:tir, no IAA
      - yap-1:aid, col-10:tir, no IAA
      - yap-1:aid, raga-1:aid, col-10:tir, no IAA
```

Where all the settings with one element are distributed across all conditions bellow them. By default, we automatically merge conditions with the same parameters (except `point_range` and `description`) across experiments.

## Example

An example Jupyter notebook is available in the `analysis_and_plots` directory of the repository. It will be kept up to date with the latest version of the code. To get started, copy the notebook and give it a different name. You can then modify it to fit your needs. This copied version will not be overwritten when we update the original notebook.
