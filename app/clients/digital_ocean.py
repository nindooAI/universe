import boto3
import progressbar

from loguru import logger


class DOClient:

    def __init__(self, CLIENT_ID, DO_URL, DO_KEY_ID, DO_SECRET):
        self.do = boto3.client('s3',
                               region_name='sfo2',
                               endpoint_url=DO_URL,
                               aws_access_key_id=DO_KEY_ID,
                               aws_secret_access_key=DO_SECRET)
        self.bucket = CLIENT_ID

    def download_callback(self, size):
        self.pg.update(self.pg.currval + size)

    def download_models(self):
        model_keys = ['PT-EN.tar.gz', 'EN-ES.tar.gz']
        for key in model_keys:
            logger.info(
                '[*] Downloading {} from DO Spaces to ./{}'.format(key, key))

            response = self.do.head_object(Bucket=self.bucket, Key=key)

            self.pg = progressbar.progressbar.ProgressBar(
                maxval=response['ContentLength']
            )
            self.pg.start()
            with open('./{}'.format(key), 'wb') as f:
                self.do.download_fileobj(
                    self.bucket, key, f, Callback=self.download_callback)

            logger.info(
                '[x] Downloaded {} from DO Spaces to ./{}'.format(key, key))
