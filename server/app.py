from openenv.core.env_server.http_server import create_app
from models import DataCleanAction, DataCleanObservation
from server.environment import DataCleanEnvironment

app = create_app(DataCleanEnvironment, DataCleanAction, DataCleanObservation)