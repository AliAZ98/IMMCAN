# Dataset placement

Dataset files and candidate images are not tracked in git. Obtain AdMIRe
Subtask A data through the organizers under the applicable license.

Example (one language):

data/
  EN/
    train/
      subtask_a_train.tsv
      <compound>/
        <candidate>.png
    dev/
      subtask_a_dev.tsv
      <compound>/
        <candidate>.png

Repeat for PT when available. TSV input requires compound, sentence,
sentence_type, expected_order, and image1..image5 name/caption columns.

The loader preserves the TSV image1..image5 candidate order, derives ranks
from expected_order, and does not feed answer order as model input.

For text-only experiments, image files are NOT loaded, but TSVs and image-root
arguments are still required so the same paired-data interface can be used.
Augmented TSVs must be paired with the correct ORIGINAL image directories;
do not pass more TSVs than root paths.
