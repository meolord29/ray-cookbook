# train via ray remote

# Purpose: 
# 
# To learn how to use basic functions of remote ray and how to use the gpu with ray

# ==========[local package imports]===========

import ray


# ==[Set up packages on the remote ray cluster]==

runtime_env = {
    "pip": {
        "packages": ["torch", "torchvision", "torchaudio", "sentence-transformers[train]", "datasets", "minio"]
    }
}

# Init the remote cluster
ray.init(address="ray://192.168.3.179:10001", runtime_env=runtime_env)


print(ray.cluster_resources())


# ===========[ Tasks ]===================

# validate usage of the gpu
@ray.remote(num_gpus=1) #num_gpus=1
def train_run():
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
    from sentence_transformers.losses import CoSENTLoss
    from datasets import load_dataset, Dataset
    import pyarrow as pa
    import pandas as pd

    from minio import Minio, S3Error
    import os
    from math import floor
    from pathlib import Path

    def upload_folder_to_minio(local_folder, bucket_name):
        # Initialize Minio client
        url = '192.168.2.64:7000'
        access_key = "ucu8sKHvX5rwCPmm"
        secret_key = "gJmstFgDRsoeGF4IcXoAhUtu2TqsEAgH"

        client = Minio(
            endpoint = url,
            access_key=access_key,
            secret_key=secret_key,
            secure=False
        )

        # Check if bucket exists, create if not
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            print(f"Created bucket {bucket_name}")
        else:
            print(f"Bucket {bucket_name} already exists")

        # Get the base folder name
        folder_name = os.path.basename(os.path.normpath(local_folder))
        
        # Walk through directory and upload files
        for root, _, files in os.walk(local_folder):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Generate object name with folder prefix
                relative_path = os.path.relpath(file_path, start=local_folder)
                object_name = f"{folder_name}/{relative_path}"
                
                # Convert Windows paths to POSIX style
                object_name = object_name.replace("\\", "/")

                # Upload the file
                client.fput_object(
                    bucket_name,
                    object_name,
                    file_path
                )
                print(f"Uploaded {file_path} to {bucket_name}/{object_name}")

    def get_object(bucket_name, item_name):
        '''
        Gets a object byte stream from a bucket.
        '''
        #logger = create_logger()

        url = '192.168.2.64:7000'
        access_key = "ucu8sKHvX5rwCPmm"
        secret_key = "gJmstFgDRsoeGF4IcXoAhUtu2TqsEAgH"

        # Get data of an object.
        try:
            # Create client with access and secret key
            client = Minio(url,  # host.docker.internal
                        access_key, 
                        secret_key,
                        secure=False)


            object_response = client.get_object(bucket_name, item_name) # returns respons in bytes, needs to be converted into an appropriate object type

        except S3Error as s3_err:
            #logger.error(f'S3 Error occurred: {s3_err}.')
            raise s3_err
        except Exception as err:
            #logger.error(f'Error occurred: {err}.')
            raise err

        return object_response

    def get_pd_dataframe(bucket = "try-v002", arrow_path = "data/stsb_multi_mt-train.arrow"):
        response = get_object(bucket, arrow_path)

        ds = ray.data.from_arrow(response.data) # reads arrow bytes into a dataset

        return ds.to_pandas()


    # 1.0  Load a model to finetune
    model = SentenceTransformer("roberta-large")
    
    # 2.0 load several Datasets to train with

    

    # (sentnece1, sentence2) + score
    df_train = get_pd_dataframe(bucket = "try-v002", arrow_path = "data/stsb_multi_mt-train.arrow") #pd.read_csv("train.csv")

    #print(df_train)

    print(df_train.columns.tolist())

    #df_train['similarity_score'] = [floor(x) for x in df_train['similarity_score']]

    df_train.rename(columns={'similarity_score': 'label'}, inplace=True)

    #df_train = df_train.drop(["Unnamed: 0"], axis = 1)

    stsb_pair_score_train = Dataset.from_pandas(df_train)

    # We can combine all datasets into a dictionary with a dataset names to datasets
    train_dataset = {
        "stsb": stsb_pair_score_train,
    }

    # 3. Load several Datasets to evaluate with
    # (sentence1, sentence2, score)

    #df_val = pd.read_csv("validate.csv")
    df_val = get_pd_dataframe(bucket = "try-v002", arrow_path = "data/stsb_multi_mt-test.arrow")

    #df_val['similarity_score'] = [floor(x) for x in df_val['similarity_score']]
    #df_val = df_val.drop(["Unnamed: 0"], axis = 1)

    df_val.rename(columns={'similarity_score': 'label'}, inplace=True)

    stsb_pair_score_dev = Dataset.from_pandas(df_val)

    # We can use dictionary for the evaluation too, but we don't have to. We could also just use
    # no evaluation dataset, or one dataset
    eval_dataset = {
        "stsb": stsb_pair_score_dev,
    }

    # 4. Load several loss functions to train with
    # (sentence_A, sentence_B) + score
    cosent_loss = CoSENTLoss(model)

    # Create a mapping with  dataset names to loss functions, so the trainer knows which loss to apply where.
    # Note that you can also just one loss if all of your training/evaluation datasets use the same loss
    losses = {
        "stsb": cosent_loss
    }

    # 5. Define a simple trainerm, although it's reccomend to use one with args & evaluators
    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=losses
    )

    trainer.train()

    # 6. save the trained model and optionally push it to huggingface face hub
    model.save_pretrained("all_datasets_v3_roberta-custom_embedding")

    print(os.listdir("."))

    upload_folder_to_minio('all_datasets_v3_roberta-custom_embedding', "try-v002")


print(ray.get(train_run.remote()))

