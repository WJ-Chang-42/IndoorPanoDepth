numbers=0
d=1.5
name=ours
SCENES="classroom bedroom kitchen livingroom loft"
mkdir ./exp/"$d"_"$name"/	
for scene in $SCENES; do
/opt/conda/bin/python -u exp_runner.py --conf ./confs/"$name".conf --case "$scene" --d "$d" --n "$numbers" --dir ./exp/"$d"_"$name"/ --random 80 | tee ./exp/"$d"_"$name"/"$numbers"_images_"$scene".txt
done

