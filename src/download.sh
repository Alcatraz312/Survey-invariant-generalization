BASE="/home/arbiter/projects/Survey-invariant-generalization/data/SDSS_B_chunks"
find "$BASE" -type f -name "*chunk*" | while read file; do
    echo "Downloading list: $file"

    wget -c \
        --no-check-certificate \
        --tries=5 \
        --timeout=30 \
        --connect-timeout=15 \
        --read-timeout=30 \
        --dns-timeout=15 \
        -nd \
        -i "$file" \
        -P "/home/arbiter/projects/Survey-invariant-generalization/data/fits_files/SDSS_B_fits"\
        -nv
done