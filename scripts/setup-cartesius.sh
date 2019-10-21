. scripts/init-cartesius.sh
virtualenv venv
. venv/bin/activate
pip install torch==1.2.0 torchvision==0.4.0 tqdm==4.36.1 \
        tensorflow-gpu==1.12.0 tensorflow-gpu==1.12.0 cleverhans==3.0.1
