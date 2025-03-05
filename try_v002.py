# Purpose: 
# 
# To learn how to use minio and and how to communicate with it


# ======== [import packages] ==========
import ray


# ==[Set up packages on the remote ray cluster]==

runtime_env = {
    "pip": {
        "packages": ["torch", "torchvision", "torchaudio", "emoji"]
    }
}

# Init the remote cluster
#ray.init(address="ray://192.168.3.179:10001", runtime_env=runtime_env)

# Make sure to install: `pip install -U s3fs`
import s3fs
import pyarrow.fs

s3_fs = s3fs.S3FileSystem(
    key='ucu8sKHvX5rwCPmm',
    secret='gJmstFgDRsoeGF4IcXoAhUtu2TqsEAgH',
    endpoint_url='http://192.168.2.64:7000/' # secondary port, differs to UI port
)

print(s3_fs.ls("try-v002")) 
# validated that this works, learned that it must be on the secondary port

#custom_fs = pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(s3_fs))



def get_object_list(bucket_name: str) -> list[str]:
   from minio import Minio, S3Error
   import os
   '''
   Gets a list of objects from a bucket.
   '''
   #logger = create_logger()

   url = '192.168.2.64:7000' # NO "http://"
   access_key = "ucu8sKHvX5rwCPmm"
   secret_key = "gJmstFgDRsoeGF4IcXoAhUtu2TqsEAgH"

   # Get data of an object.
   try:
       # Create client with access and secret key
       client = Minio(url,  # host.docker.internal
                   access_key, 
                   secret_key,
                   secure=False)

       object_list = []
       objects = client.list_objects(bucket_name, recursive=True)
       for obj in objects:
           object_list.append(obj.object_name)
   except S3Error as s3_err:
       #logger.error(f'S3 Error occurred: {s3_err}.')
       raise s3_err
   except Exception as err:
       #logger.error(f'Error occurred: {err}.')
       raise err

   return object_list


objects = get_object_list("try-v002")
print(objects)


def get_object(bucket_name, item_name):

    from minio import Minio, S3Error
    import os
    '''
    Gets a list of objects from a bucket.
    '''
    #logger = create_logger()

    url = '192.168.2.64:7000' # NO "http://"
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


response = get_object("try-v002", "data/stsb_multi_mt-train.arrow")

#print(response.data)


ds = ray.data.from_arrow(response.data) # reads arrow bytes into a dataset
print(ds)

# import pyarrow as pa
# buf = memoryview(b"some data")
# with pa.input_stream(buf) as stream:
#     stream.read(4)