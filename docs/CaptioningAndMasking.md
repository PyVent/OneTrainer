# Captioning and Masking

## Using captions in training

Open a concept and select **Captions**. Captions can come from a UTF-8 TXT next to
each image, one shared TXT, or the image filename. In a TXT, each non-empty line
is a complete caption.

**Every line is a separate example** trains an image once for each caption. For
example, one image with a tag line and a prose line produces two examples. Image
and text augmentation run independently for each example. A missing or empty TXT
keeps one example with an empty caption. A shared TXT applies all its lines to
every image.

**Pick one random line** retains the previous behavior: one caption is selected
per text variation. Existing concepts keep this mode and their tag processing
rules. Newly created concepts default to separate examples and mixed text.

For mixed captions, **Mixed: detect tags or prose** checks each line independently;
their order in the file does not matter. Detection is approximate. Uncertain
lines are treated as prose. In the preview, choose a line and override its type
when necessary. Overrides are saved with the concept and match the exact caption
text, so reordering lines preserves them; editing that text requires a new override.
Tag shuffling, removal and capitalization apply only to lines classified as tags.

**Sampling** controls repeats and cached augmentation variants. Two examples with
two repeats produce four samples before batching. Cached variants do not multiply
the epoch size: they are reused across repetitions. With caching disabled, each
repetition gets fresh augmentation. Changing caption contents invalidates the
text cache; changing the number of separate examples also invalidates the image
cache to preserve image/caption alignment.

The editor applies changes on **Save concept**. **Cancel** discards them. Opening
the settings does not scan the dataset. The preview reads the selected image and
its caption file, discovering subsequent images on demand. **Checks** explicitly
counts the entire dataset or inspects its metadata; scans can be cancelled.
Counts update while scanning, including within a large directory.

### Preparation progress and the caption index

Training reports file enumeration, caption indexing, cache preparation and aspect
bucket sorting separately. Random-caption mode opens TXT files only when needed.
Separate-example mode needs the number of non-empty lines before the epoch starts.
It stores these counts in `<cache_dir>/caption_index`, checks file sizes and
modification times from directory enumeration, and reads only new or changed TXT
files. The first count uses a bounded pool of eight readers. It keeps counts, not
the entire dataset's caption text, in memory. Missing or corrupt indexes rebuild
automatically. Delete this directory to force a recount if files were edited while
deliberately preserving both their size and modification time.

## Captioning and masking tools

OneTrainer includes a UI for captioning and masking of your dataset. To access it, click on `Dataset Tools` within the `tools`
tab. Once the UI is open, click the `Open` button at the top left, and select the directory of your dataset.

### Navigating the UI

To switch between the images, either click on the filename in the list on the left side, or use the up and down arrow
keys to go to the next or previous image.

### Manual captioning

In the input box at the bottom you can input your caption. To save the caption, press enter.

### Automatic captioning

Clicking on the `Generate Captions` button opens the batch captioning modal. Here you can choose which model to use for
captioning. For the initial caption, you can choose a text that should be used to start the new caption. To generate the
captions, press the `Create Captions` button. When you use this for the first time, it has to download the model.
Depending on the model you chose, this can take a while.

Entries in the `Prefix` field will be added at the start of the caption. `Postfix` will be added at the end of the caption.

### Manual masking

Check the `Enable Mask Editing` checkbox at the top. Now you can draw a mask onto the image. Left-click adds to the
masked region, right click removes parts from the mask. With the mouse wheel you can increase or decrease the brush
size. Use `Ctrl + M` to only show the mask. To save the mask, click into the caption input field, then press enter.

### Automatic masking

Clicking on the `Generate Masks` button opens the batch masking modal. Here you can choose which model to use for
masking. Some models like ClipSeg support masking based on a prompt. Play around with Threshold, Smooth and Expand values to find what works best for your dataset.

To generate the masks, press the "Create Masks" button at the bottom. When you use this for the first time, it has to download the model. Depending on the model you chose, this can take a while.
