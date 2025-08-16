while getopts g:n:p: flag
do
        case "${flag}" in
                g)  gpu=${OPTARG};;
                n)  number=${OPTARG};;
                p)  port=${OPTARG};;
        esac
done
echo "Running container spoof_$number on gpu $gpu and port $port";
docker run --rm -it --gpus '"device='$gpu'"' --userns=host --shm-size 64G -v $PWD:/workspace/spoof/ -p $port --name spoof_$number spoof:latest
