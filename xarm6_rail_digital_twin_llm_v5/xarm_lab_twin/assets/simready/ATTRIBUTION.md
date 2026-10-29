# Attribution

The assets in this folder are converted from NVIDIA SimReady assets, published
in the Hugging Face dataset
[`nvidia/simready-assets`](https://huggingface.co/datasets/nvidia/simready-assets)
under the [Creative Commons Attribution 4.0 International licence
(CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/). Copyright NVIDIA
Corporation.

| slug | source asset (revision a666cb4381ba513b174fb120ed523c1566ba832c) |
|---|---|
| `bottled_drink_e01` | `food_beverage_bottledDrink_e01/simready_usd/sm_food_beverage_bottledDrink_e01_01.usd` |
| `cucumber_a01` | `food_produce_cucumber_a01/simready_usd/sm_food_produce_cucumber_a01_01.usda` |
| `cupcake_g01` | `food_prepared_cupcake_g01/simready_usd/sm_food_prepared_cupcake_g01_01.usd` |

**Changes made** by `simready_import`:
- converted from OpenUSD to MuJoCo MJCF with OBJ meshes;
- fixed-jointed parts merged into one rigid body;
- the origin moved to the bounding-box centre;
- convex collision hulls computed;
- textures downscaled to at most 1024 px;
- rolling and torsional friction added.

`manifest.json` records the exact source file, revision and a sha256 of every
converted file.
