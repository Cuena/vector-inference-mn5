# Configs

* [`environment.yaml`](environment.yaml): Configuration for the Slurm cluster environment, including image paths, resource availabilities, default value, and etc.
* [`models.yaml`](models.yaml): Configuration for launching model inference servers, including Slurm parameters as well as inference engine arguments.

**NOTE**: These configs act as last-resort fallbacks in the `vec-inf` package. They are updated to match the latest cached config on the Vector Killarney cluster with each new package version release.

* [`marenostrum5/`](marenostrum5): Optional MareNostrum5-specific profile (environment + model overrides).
