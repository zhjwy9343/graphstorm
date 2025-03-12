# build an inference image
bash ./build_gsf_test.sh -i 911734752298.dkr.ecr.us-east-1.amazonaws.com/graphstorm-james -e sagemaker -d cpu -s -infer

# create an ECR repository

# authenticate ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 911734752298.dkr.ecr.us-east-1.amazonaws.com

# push the image up
docker push 911734752298.dkr.ecr.us-east-1.amazonaws.com/graphstorm-james:sagemaker-cpu-infer


# run docker locally
docker run -it -p 8080:8080 \
  -v /path/to/local/code:/opt/ml/code \
  -v /path/to/local/model:/opt/ml/model \
  -e SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code \
  -e SAGEMAKER_PROGRAM=infer_entry_point.py \
  911734752298.dkr.ecr.us-east-1.amazonaws.com/graphstorm-james:sagemaker-cpu-infer /bin/bash

docker run -it -p 8080:8080 \
  911734752298.dkr.ecr.us-east-1.amazonaws.com/graphstorm-james:sagemaker-cpu-infer /bin/bash

docker run -p 8080:8080 \
  911734752298.dkr.ecr.us-east-1.amazonaws.com/graphstorm-james:sagemaker-cpu-infer

docker run -it -p 8080:8080 \
  -v code:/opt/ml/model \
  911734752298.dkr.ecr.us-east-1.amazonaws.com/graphstorm-james:sagemaker-cpu-infer /bin/bash

docker container exec -it bfae337ec6f4 /bin/bash



