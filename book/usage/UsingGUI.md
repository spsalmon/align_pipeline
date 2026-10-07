# Using the GUI

To access the GUI, first modify the **FILEMAP_PATH** variable in `gui/launch_gui.sh` to point to the filemap you want to open. Then run the following commands in your terminal:

```bash
bash gui/launch_gui.sh
```

You can also launch it directly, without editing the script:

```bash
python gui/run.py --filemap /path/to/analysis_filemap.parquet
```

When opening a filemap in the GUI, we automatically create a backup in the same folder. The GUI then saves the annotated version of the filemap with an "_annotated" suffix. By default, the GUI always tries to load the annotated version if it exists. You can disable this behavior by switching **OPEN_ANNOTATED** to 0 in `gui/launch_gui.sh`. If set to 0, it will always open exactly the filemap path provided. You may chose to have the GUI recompute all values at molt on launch by setting **RECOMPUTE_VALUES_AT_MOLT** to 1.

The Shiny GUI is a way to verify, annotate and explore your data in a more interactive way. It is meant to be used after all the parts of your pipeline finished running. The plot shows the measurement across time of the selected individual. You can select the feature you want to plot on the y axis using **Select column to plot**. Data points corresponding to errors as detected through the automated quality control appear as triangles. Eggs appear as squares and the rest as circles. You can click on a data point to jump to the corresponding time and image. You can select which image channel is being displayed and overlay segmentation masks using the dropdown menues on the right.

The GUI lets you see the molts detected by the automatic molt detection algorithm. Clicking one of the ecdysis buttons will annotate the current time as this molt exit. Doing so automatically interpolates the values of each series at the time of the molt (shown by a cross of the corresponding color). If you are unhappy with this interpolation, clicking **Select value at molt** will peg it to the values at the current time.

You can annotate special events using custom columns. If you type the name of a custom column in the text input, the next time you click **Annotate** will add it to the dataframe. You can then select which column you are annotating using the dropdown menu.

If for some reason you already have annotations for this experiment in another analysis filemap, you can import them to the current one using the **Import annotations** button. With **Recompute values at molt** ticked (the default), only the annotations (hatch, molts, Death, Ignore, Arrest, custom columns) are taken from the imported file, and all values at ecdysis are recomputed from the features and QC of the filemap currently open in the GUI. Untick it to keep the values at ecdysis stored in the imported file and only fill in the missing ones. Either way, events that have no time in the imported file end up with no value.
