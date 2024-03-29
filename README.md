# Panoptes model in tf2

Model abbreviations:
- X1: Panoptes2, InceptionResNetV2 X 3, without feature pooling
    - with abbreviations: `--variant=X1`
    - with flags: `--base_model='InceptionResNetV2' --feature_pool=False`

- X2: Panoptes1, InceptionResNetV1 X 3, without feature pooling
    - with abbreviations: `--variant=X2`
    - flags: `--base_model='InceptionResNetV1' --feature_pool=False`

- X3: Panoptes4, InceptionResNetV2 X 3, with feature pooling
    - with abbreviations: `--variant=X3`
    - flags: `--base_model='InceptionResNetV2' --feature_pool=True`

- X4: Panoptes3, InceptionResNetV1 X 3, with feature pooling
    - with abbreviations: `--variant=X4`
    - flags: `--base_model='InceptionResNetV1' --feature_pool=True`

- F{1, 2, 3, 4}: X{1, 2, 3, 4} with covariates (age, BMI)
    - with abbreviations: `--variant=F{1, 2 ,3, 4}`
    - additonal flag: `--covariate=age,BMI`

- I5: Single-resolution InceptionResNetv1

- I6: Single-resolution InceptionResNetv2

Use the corresponding command line arguments to instantiate the desired Panoptes variant.

See [examples](scripts/cli.sh) for more details.