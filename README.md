# movenet-api

tensorflow(2.13.0) の [movenet](https://tfhub.dev/google/movenet/multipose/lightning) を使った姿勢推定を行う RestAPI

## build

### for CPU

```bash
docker build . -t movenet-api
```

### for GPU (CUDA)

```bash
docker build . -t movenet-api -f ./dockerfile.gpu 
```

## Run

### for CPU

```bash
docker run -d -p 5000:5000 -v ./tfhub_modules:/tmp/tfhub_modules --name movenet-api movenet-api 
```

### for GPU (CUDA)

```bash
docker run --gpus all -d -p 5000:5000 -v ./tfhub_modules:/tmp/tfhub_modules --name movenet-api movenet-api 
```

## Test

### request json
> {
>     "image": "... base64 encoded jpeg binary ..."
> }

### send post
```
curl -X POST -H "Content-Type: application/json" http://localhost:5000/api/inference/ -d "@request.json" -v --output inference.jpg
```
