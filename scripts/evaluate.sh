python scripts/evaluate.py -m cnn -o table
for m in 'RN50' 'RN101' 'RN50x4' 'RN50x16' 'ViT-B-32' 'ViT-B-16'; do
    for t in word photo-of; do
        python scripts/evaluate.py -m clip-${m}-${t} -o table
    done
done
python scripts/evaluate.py -m cnnattend-soft -o table
