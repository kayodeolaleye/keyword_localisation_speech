python scripts/evaluate_visual_vs_text.py -vm cnn -o table
for m in 'RN50' 'RN101' 'RN50x4' 'RN50x16' 'ViT-B-32' 'ViT-B-16'; do
    for t in word photo-of; do
        python scripts/evaluate_visual_vs_text.py -vm clip-${m}-${t} -o table
    done
done
