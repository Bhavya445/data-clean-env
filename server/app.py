from openenv.core.env_server.http_server import create_app
from models import DataCleanAction, DataCleanObservation
from server.environment import DataCleanEnvironment

app = create_app(DataCleanEnvironment, DataCleanAction, DataCleanObservation)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()