from src.utils import get_env_var

broker_url = get_env_var('CELERY_BROKER_URL')
result_backend = get_env_var('CELERY_RESULT_BACKEND')
