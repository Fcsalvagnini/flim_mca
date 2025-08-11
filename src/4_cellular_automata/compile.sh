if [ "$#" -ne 1 ]; then
  echo "Usage: sh $0 (0: CPU, 1: GPU)" >&2
  exit 1
fi

if [ "$1" -eq 0 ]; then
  echo "Compiling CPU version..."
  make IFT_GPU=0 iftCA
elif [ "$1" -eq 1 ]; then
  echo "Compiling GPU version..."
  make IFT_GPU=1 iftCAGpu
else
  echo "Error: Argument must be 0 (CPU) or 1 (GPU)" >&2
  exit 1
fi