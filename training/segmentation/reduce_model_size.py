import os

from align_toolbox.deep_learning.utils.util import create_lightweight_checkpoint

if __name__ == "__main__":
    model_dir = "/mnt/towbin.data/shared/spsalmon/towbinlab_segmentation_database/models/paper/body/towbintools_medium"
    model_name = "epoch=885-step=52274.ckpt"
    output_name = "best_light.ckpt"

    create_lightweight_checkpoint(
        input_path=os.path.join(model_dir, model_name),
        output_path=os.path.join(model_dir, output_name),
    )
