numbers=0
d=1.5
name=Stanford2D3D
SCENES="1_area_5a1 1_area_5b1 5_area_5a1 10_area_61 207_area_41"	
mkdir ./exp/"$d"_"$name"/	
for scene in $SCENES; do
/opt/conda/bin/python -u exp_runner.py --conf ./confs/"$name".conf --case "$scene" --d "$d" --n "$numbers" --dir ./exp/"$d"_"$name"/ --random 80 | tee ./exp/"$d"_"$name"/"$numbers"_images_"$scene".txt
done

