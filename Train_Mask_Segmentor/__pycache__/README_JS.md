# Segment Anything Model


- **[Medical-SAM-Adapter](https://github.com/KidsWithTokens/Medical-SAM-Adapter)**

[https://github.com/KidsWithTokens/Medical-SAM-Adapter](https://github.com/KidsWithTokens/Medical-SAM-Adapter)

- Edited File list

Medical-SAM-Adapter/models/sam/build_sam.py

Medical-SAM-Adapter/models/sam/modeling/prompt_encoder.py

Medical-SAM-Adapter/models/sam/modeling/mask_decoder.py

Medical-SAM-Adapter/models/sam/modeling/text_encoder.py

Medical-SAM-Adapter/models/sam/modeling/sam.py

Medical-SAM-Adapter/function.py

Medical-SAM-Adapter/cfg.py

Medical-SAM-Adapter/dataset.py

Medical-SAM-Adapter/utils.py

Medical-SAM-Adapter/train.py

If direct modifications to the model are necessary or if a deeper understanding of the model is required, it is recommended to focus on reviewing the respective files.

- Train

You need to create your own Dataset Class in [dataset.py](http://dataset.py/) and modify the [function.py](http://function.py/) file accordingly to fit your dataset. Training can be initiated by running [train.py](http://train.py/)

Additionally, you need to check the [cfg.py](http://cfg.py/) file and modify the dataset path and model checkpoint according to your environment.