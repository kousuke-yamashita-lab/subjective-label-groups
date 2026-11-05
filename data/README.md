# Data

**Status: not yet released.** The files below are described so that the analysis code can be read.

Survey images and image features are not distributed. The survey images were collected from the web without permission from the rights holders, and image content can be partly inferred from such features. Raters and images are identified only by arbitrary IDs. Dates and demographic attributes are not included.

| File | Contents |
|---|---|
| `ratings.csv` | One row per rater and image: `rater_id`, `group` (JP or VN), `set_id` (S1 to S6), `condition`, `image_id`, `overall` (0 to 100), and six elements on a 5-point scale (`color`, `pattern`, `silhouette`, `material`, `accessory`, `styling`) |
| `selection_scores.csv` | The score of the pre-trained model used to build the image sets: `image_id`, `set_id`, `selection_score` |
| `train_ratings.csv` | The separate Japanese ratings used to train the transfer model: `rater_id`, `form_id`, `image_id`, `overall` |
| `model_predictions.csv` | Per-image predictions of the transfer model, one column per image feature (`siglip2_base`, `siglip_so400m`, `clip_l14`, `fashion_clip`, `vit_in21k`) |
| `precomputed_model_results.json` | Results that need the image features, fixed from a single full run: the two leave-one-set-out comparisons and the model gap interval |
