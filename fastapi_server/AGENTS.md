# Backend Development Instructions

The agent application template includes a backend implementation in `fastapi_server/`.
By default it ships a backend implementing APIs endpoints for the frontend application.

## Backend Development Guidelines

- The FastAPI backend in `fastapi_server/` already serves the chat API at `/api/v1/`.
  If the user's frontend needs new data endpoints, add them in `fastapi_server/app/api/v1/`.
- The entry point for the backend can be found at `fastapi_server/app/main.py`
- For POST endpoints accepting JSON body, use Pydantic models (not function parameters). Query params go in function signature, body params go in Pydantic model.

## Installing backend packages

Before making any changes to the backen code, install dependencies by running shell command:

```shell
dr task run fastapi_server:install
```

## Backend Testing

```shell
dr task run fastapi_server:lint
```

```shell
dr task run fastapi_server:test
```

