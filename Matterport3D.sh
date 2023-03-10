numbers=0
d=1.5
name=Matterport3D
SCENES="0_0b217f59904d4bdf85d35da2cab963471 1_0b724f78b3c04feeb3e744945517073d1 0_a2577698031844e7a5982c8ee0fecdeb1 0_9f2deaf4cf954d7aa43ce5dc70e7abbe1 0_7812e14df5e746388ff6cfe8b043950a1 4_0b724f78b3c04feeb3e744945517073d1 2_0b217f59904d4bdf85d35da2cab963471 1_7812e14df5e746388ff6cfe8b043950a1 47_a2577698031844e7a5982c8ee0fecdeb1 45_a2577698031844e7a5982c8ee0fecdeb1"
mkdir ./exp/"$d"_"$name"/	
for scene in $SCENES; do
/opt/conda/bin/python -u exp_runner.py --conf ./confs/"$name".conf --case "$scene" --d "$d" --n "$numbers" --dir ./exp/"$d"_"$name"/ --random 80 | tee ./exp/"$d"_"$name"/"$numbers"_images_"$scene".txt
done

