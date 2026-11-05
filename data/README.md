# Data

**Status: not yet released.** The files below are described so that the analysis code can be read. Their release is awaiting confirmation.

Survey images are not distributed, because the rights holders have not given permission. Raters and images are identified only by arbitrary IDs. Dates and demographic attributes are not included.

| File | Contents |
|---|---|
| `ratings.csv` | One row per rater and image: `rater_id`, `group` (JP or VN), `set_id` (S1 to S6), `condition`, `image_id`, `overall` (0 to 100), and six elements on a 5-point scale (`color`, `pattern`, `silhouette`, `material`, `accessory`, `styling`) |
| `selection_scores.csv` | The score of the pre-trained model used to build the image sets: `image_id`, `set_id`, `selection_score` |
| `train_ratings.csv` | The separate Japanese ratings used to train the transfer model: `rater_id`, `form_id`, `image_id`, `overall` |
| `features_<model>.npz` | Frozen image features: `image_ids`, `X` for the 60 survey images, and `train_image_ids`, `train_X` for the training images |

Feature files are provided for `siglip2_base`, `siglip_so400m`, `clip_l14`, `fashion_clip`, and `vit_in21k`.
