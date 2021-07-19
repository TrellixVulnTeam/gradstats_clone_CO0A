To build customized container image with training code and additional libraries: 
	
	bash build_dlc_sagemaker.sh <image_name> <tag_name> <aws_account_id>	

Push the customized docker image to ECR:
	
	bash push_dlc_sagemaker.sh <image_name> <tag_name> <aws_account_id>	
	
Example:
	
	bash build_dlc_sagemaker.sh sagemaker test 12345
	bash push_dlc_sagemaker.sh sagemaker test 12345

After push the image to your own ECR, example command to run:
	
	pip install sagemaker
	python launch.py --num_nodes 1

