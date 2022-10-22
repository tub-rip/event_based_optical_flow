# Folder structure

The folder structure should follow below:

```shell
tree -L 3
.
├── MVSEC
│   ├── hdf5
│       ├── indoor_flying1_data.hdf5
│           ...
│   └── gt_flow
│       ├── indoor_flying1_gt_flow_dist.npz
│           ...
└── README.md  # this readme 
```

Please download datasets accordingly.

- MVSEC data from (https://drive.google.com/drive/folders/1gDy2PwVOu_FPOsEZjojdWEB2ZHmpio8D)

# Your own dataset location

Optionally, you don't have to locate the files here,
rather choose your dataset root directory location.

You need to specify the root folder with the config yaml file.

