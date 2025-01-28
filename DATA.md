## Data Preparation

Please download videos of the dataset first, and link to the path through symbolic link. The instructions for each dataset are provided below.

### JHMDB

1. make folder for symbolic links
```bash!
cd ST-CLIP/ZS_JHMDB/ST-CLIP
mkdir data/jhmdb
```

2. download videos (frames with .png format) from [official dataset website](http://jhmdb.is.tue.mpg.de/challenge/JHMDB/datasets)
3. make symbolic links in `data/jhmdb`
```bash
ln -s 'path/to/videos' 'data/jhmdb/videos'
ln -s 'ST-CLIP/ZS_JHMDB/annotation/jhmdb_train_gt_min.json' 'data/jhmdb/annotations/jhmdb_train_gt_min.json'
ln -s 'ST-CLIP/ZS_JHMDB/annotation/jhmdb_test_gt_min.json' 'data/jhmdb/annotations/jhmdb_test_gt_min.json'
ln -s 'ST-CLIP/ZS_JHMDB/annotation/all_label.txt' 'data/jhmdb/annotations/all_label.txt'
ln -s 'ST-CLIP/ZS_JHMDB/annotation/testlist.txt' 'data/jhmdb/annotations/testlist.txt'
ln -s 'ST-CLIP/ZS_JHMDB/annotation/jhmdb_faster_RCNN_person.json' 'data/jhmdb/jhmdb_faster_RCNN_person.json'
```