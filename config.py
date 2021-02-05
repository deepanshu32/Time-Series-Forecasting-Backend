import os
from dotenv import load_dotenv


class BaseConfig(object):
    """ load dotenv in the base root
     refers to application_top """
    APP_ROOT = os.path.join(os.path.dirname(__file__), '..')
    dotenv_path = os.path.join(APP_ROOT, '.env')
    load_dotenv(dotenv_path)

    LOCAL_FILE_PATH = os.getenv('LOCAL_FILE_PATH')
    # s3 code starts
    BUCKET_NAME = os.getenv('BUCKET_NAME')
    AWS_ACCESS_KEY_ID = os.getenv('S3_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('S3_SECRET_ACCESS_KEY')
    MONGO_DBNAME = os.getenv('MONGO_DBNAME') 
    MONGO_URI = os.getenv('MONGO_URI') 
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY') 