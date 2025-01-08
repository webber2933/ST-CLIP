## Data Preparation

Please download some videos and annotations of the dataset first, and link to these paths through symbolic links.

### JHMDB

1. make folder for symbolic links
```bash
mkdir data/jhmdb
```

2. download videos (frames with .png format) from [official dataset website](http://jhmdb.is.tue.mpg.de/challenge/JHMDB/datasets)
3. download annotation files 
(1) training data annotation [jhmdb_train_gt_min.json](https://drive.google.com/file/d/1Xt6A8f-9zgGHXtZbNPwiNuzVkQu_f8P4/view?usp=drive_link)
(2) testing data annotation [jhmdb_test_gt_min.json](https://drive.google.com/file/d/13CV1Uhxcq-PmjSBwNN3kyBmKpvSJpt5Y/view?usp=drive_link)
(3) person detection [jhmdb_faster_RCNN_person.json](https://drive.google.com/file/d/1Vfzl8HTZL7YRX5e-4CziQ_u7ctXzZPFC/view?usp=drive_link)
(4) all label names [all_label.txt](https://drive.google.com/file/d/1GiKTZMOzdS14m-HMd5EmaD72Jtx6bEMA/view?usp=drive_link)
(5) testlist [testlist.txt](https://drive.google.com/file/d/19zWUapdXLRRakQf5O5q4JkdRzEm6MJAx/view?usp=drive_link)
4. make symbolic links in `data/jhmdb`
```bash
ln -s 'path/to/videos' 'data/jhmdb/videos'
ln -s 'path/to/jhmdb_train_gt_min.json' 'data/jhmdb/annotations/jhmdb_train_gt_min.json'
ln -s 'path/to/jhmdb_test_gt_min.json' 'data/jhmdb/annotations/jhmdb_test_gt_min.json'
ln -s 'path/to/all_label.txt' 'data/jhmdb/annotations/all_label.txt'
ln -s 'path/to/testlist.txt' 'data/jhmdb/annotations/testlist.txt'
```
5. put person detection file in `data/jhmdb`