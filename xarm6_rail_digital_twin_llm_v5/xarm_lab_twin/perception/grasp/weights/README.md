# GG-CNN weights

`state_dict`s converted from UFACTORY's pickled whole-model files.

| file | net | params | upstream file |
|---|---|---|---|
| `ggcnn_epoch_23_cornell.pt` | GGCNN | 62,420 | `ggcnn_grasping_demo/models/ggcnn_epoch_23_cornell` |
| `ggcnn2_epoch_50_cornell.pt` | GGCNN2 | 66,676 | `ggcnn_grasping_demo/models/epoch_50_cornell` |

Trained on the Cornell Grasping Dataset by Douglas Morrison (ACRV-QUT), BSD-3.
See `../LICENSE.ggcnn` and `../LICENSE.ufactory`.

Regenerate or re-verify:

```bash
python -m perception.grasp.convert_weights --verify-only   # check, write nothing
python -m perception.grasp.convert_weights --src ~/Models/ufactory_vision
```

Both models reproduce upstream's output bit-for-bit (max difference 0.0 on
random input). See `../README.md` for why they were converted rather than
vendored as-is — briefly: upstream's format requires `weights_only=False`, which
executes code from the file, and pins an importable `models` package path.
