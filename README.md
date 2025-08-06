# Kausal Paths

Kausal Paths is a tool for predicting the future emissions of cities based on historical emission data and various climate actions. Users can compare how emissions develop in different scenarios. Data is provided using a GraphQL API.

## Installation

### Development

In the project root directory, create and activate a Python virtual environment:

```shell
uv venv
source .venv/bin/activate
```

Install the required Python packages:

```shell
uv sync
```

If you have access to the Kausal private extensions, you should configure the PyPI index URL in your `.envrc` file:

```shell
export UV_INDEX_KAUSAL_USERNAME=...
export UV_INDEX_KAUSAL_PASSWORD=...
```

Then install the dependencies like this:

```shell
uv sync --extra kausal
```

If you need to run Jupyter notebooks, include the `notebook` dependency group:

```shell
uv sync --group notebook --extra kausal
```

> _Note for macOS users: If you run into issues installing python-snappy, install it separately first_
>
> ```
> brew install snappy
> CPPFLAGS="-I/opt/homebrew/include -L/opt/homebrew/lib" pip install python-snappy
> ```

Create a file called `local_settings.py` in your repository root with the following contents:

```python
from paths.settings import BASE_DIR

DEBUG = True

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'paths',
        'ATOMIC_REQUESTS': True,
    }
}
```

Run migrations:

```shell
python manage.py migrate
```

Create a superuser:

> You might need the following translations during the createsuperuser operation: käyttäjätunnus = username, sähköpostiosoite = e-mail

```shell
python manage.py createsuperuser
```

Compile the translation files:

```shell
python manage.py compilemessages
```

You can now run the backend:

```shell
python manage.py runserver
```

The GraphQL API is now available at `http://127.0.0.1:8000/v1/graphql/`.
